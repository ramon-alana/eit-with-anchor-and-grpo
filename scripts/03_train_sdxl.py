# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [MIT License]
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [MIT License], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.
import contextlib
import csv
import datetime
import os
import time
from collections import defaultdict
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from absl import app, flags
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from ml_collections import config_flags
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from trl.models.sd_utils import convert_state_dict_to_diffusers

from eit_with_anchor_and_grpo.models.eit import (encode_prompts_sdxl,
                                                 load_emo_model)
from eit_with_anchor_and_grpo.models.pipeline_with_logprob import (
    ddim_step_with_logprob, sdxlpipeline_with_logprob)
from eit_with_anchor_and_grpo.models.rewards import MutiRewardModel

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "dev/dance_grpo/config/base.py", "Training configuration.")

logger = get_logger(__name__)


class PromptDataset(Dataset):
    def __init__(self, csv_path):
        self.data = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                item = {"neutral_prompt": row["Neutral_Prompt"], "arousal": float(row["Arousal"]), "valence": float(row["Valence"]), "emotional_prompt": row["Emotional_Prompt"]}
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {"neutral_prompt": item["neutral_prompt"], "arousal": torch.tensor(item["arousal"], dtype=torch.float), "valence": torch.tensor(item["valence"], dtype=torch.float), "emotional_prompt": item["emotional_prompt"]}


def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    # debug
    # print(config)
    # return

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(config.resume_from, sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1])

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int((config.sample.num_steps - 1) * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(project_dir=os.path.join(config.logdir, config.run_name), automatic_checkpoint_naming=True, total_limit=config.num_checkpoint_limit)

    if config.resume_from:
        # accelerator_config.iteration  = int(int(config.resume_from.split("_")[-1]) * int(config.save_freq)) + 1
        accelerator_config.iteration = int(config.resume_from.split("_")[-1]) + 1

    accelerator = Accelerator(
        # log_with="wandb",
        log_with="tensorboard",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(project_name="grpo-sdxl")
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        # config.pretrained.model, revision=config.pretrained.revision
        config.pretrained.model,
        # emoticrafter
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    emo_model = load_emo_model(config.emo_config_path, config.emo_ckpt_path).to(accelerator.device)
    # TODO: 1 unfrozen
    emo_model.eval()
    emo_model.requires_grad_(False)
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(position=1, disable=not accelerator.is_local_main_process, leave=False, desc="Timestep", dynamic_ncols=True)
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    emo_model = emo_model.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        target_modules = ["add_k_proj", "add_q_proj", "add_v_proj", "to_add_out", "to_k", "to_out.0", "to_q", "to_v"]
        # target_modules=["to_k", "to_q", "to_v", "to_out.0"]
        r = 32  # 4
        lora_alpha = 64  # 4
        lora_config = LoraConfig(r=r, lora_alpha=lora_alpha, init_lora_weights="gaussian", target_modules=target_modules)
        if not config.resume_from:
            pipeline.unet.add_adapter(lora_config)
        else:
            pipeline.load_lora_weights(config.resume_from, weight_name="pytorch_lora_weights.safetensors", adapter_name="default")

    # ------------
    #     显存优化
    # 1. 开启梯度检查点 (非常重要，显著减少训练时激活值显存)
    pipeline.unet.enable_gradient_checkpointing()
    # 2. 开启 VAE Slicing (减少图片解码显存)
    pipeline.enable_vae_slicing()
    # 3. 如果安装了 xformers，开启内存高效注意力 (推荐)
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass

    # ------------
    unet = pipeline.unet
    unet.to(accelerator.device, dtype=inference_dtype)
    # set up diffusers-friendly checkpoint saving with Accelerate

    def gather_tensor(tensor):
        if not dist.is_initialized():
            return tensor
        world_size = dist.get_world_size()
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, tensor)
        return torch.cat(gathered_tensors, dim=0)

    # Enable TF32 for faster training on Ampere GPUs,
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`")

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(unet.parameters() if not config.use_lora else filter(lambda p: p.requires_grad, unet.parameters()), lr=config.train.learning_rate, betas=(config.train.adam_beta1, config.train.adam_beta2), weight_decay=config.train.adam_weight_decay, eps=config.train.adam_epsilon)
    reward_model = MutiRewardModel(config.arousal_ckpt, config.valence_ckpt, device=str(accelerator.device), w_valence=config.w_valence, w_arousal=config.w_arousal, w_clip=getattr(config, "w_clip", 1), reward_mode=getattr(config, "reward_mode", "all"))  # ("va", "clip", or "all")

    # prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)

    # for sdxl
    target_dtype = pipeline.text_encoder_2.dtype if pipeline.text_encoder_2 is not None else pipeline.unet.dtype
    neg_prompt_embed = torch.zeros(1, 77, 2048, dtype=target_dtype, device=accelerator.device)  # CLIP-L (768) + OpenCLIP-G (1280) concat
    neg_pooled_embed = torch.zeros(1, 1280, dtype=target_dtype, device=accelerator.device)  # 仅 text_encoder_2 的 pooled output

    # sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    # sample_neg_pooled_embeds = neg_pooled_embed.repeat(config.train.batch_size, 1, 1)
    train_neg_pooled_embeds = neg_pooled_embed.repeat(config.train.batch_size, 1)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Prepare everything with our `accelerator`.
    unet, optimizer = accelerator.prepare(unet, optimizer)

    # Train!
    samples_per_epoch = config.sample.batch_size * accelerator.num_processes * config.num_generations
    total_train_batch_size = config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs} Recomend Full Epoches: {40000 // (config.sample.batch_size * accelerator.num_processes)}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    import torch.nn.functional as F

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0

    dataset = PromptDataset("data/prompt_mapping.csv")  # ("./assets/prompts.txt")

    sampler = DistributedSampler(dataset, num_replicas=torch.distributed.get_world_size(), rank=torch.distributed.get_rank(), seed=123543, shuffle=True)

    loader = DataLoader(dataset, batch_size=config.sample.batch_size, sampler=sampler, pin_memory=True, drop_last=True)

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
        # return
    else:
        first_epoch = 0

    global_step = 0
    for epoch, prompts in enumerate(loader):
        if epoch < first_epoch:
            continue
        #################### SAMPLING ####################
        pipeline.unet.eval()
        samples = []

        if epoch > config.num_epochs:
            break

        # ----------------------------------------------------
        # expanded_prompts = []
        # for p in prompts:
        #     expanded_prompts.extend([p] * config.num_generations)
        neutral_prompts = prompts["neutral_prompt"]  # list of str
        arousals = prompts["arousal"]  # [B] tensor
        valences = prompts["valence"]  # [B] tensor
        emotional_prompts = prompts["emotional_prompt"]  # list of str

        # 按 num_generations 扩展（针对 GRPO 的 G 次采样）
        expanded_neutral = []
        expanded_emotional = []
        expanded_arousal = []
        expanded_valence = []
        for n, a, v, e in zip(neutral_prompts, arousals, valences, emotional_prompts):
            expanded_neutral.extend([n] * config.num_generations)
            expanded_arousal.extend([a] * config.num_generations)
            expanded_valence.extend([v] * config.num_generations)
            expanded_emotional.extend([e] * config.num_generations)

        # 转换为 tensor 并堆叠
        expanded_arousal = torch.stack(expanded_arousal).to(accelerator.device, non_blocking=True)
        expanded_valence = torch.stack(expanded_valence).to(accelerator.device, non_blocking=True)
        # ----------------------------------------------------

        all_latents = []
        all_log_probs = []
        all_rewards = []
        all_prompts_embed = []
        all_pooled_prompts_embed = []

        ###for the sake of convenience, we use the same latents for all prompts in a batch.
        global_input_latents = torch.randn((1, 4, 128, 128), device=accelerator.device, dtype=inference_dtype)  # torch.bfloat16  # 64 in sdv1.4

        batch_size = config.train.batch_size
        for i in range(0, len(expanded_neutral), batch_size):
            current_batch_n = expanded_neutral[i : i + batch_size]
            current_batch_a = expanded_arousal[i : i + batch_size].unsqueeze(1)
            current_batch_v = expanded_valence[i : i + batch_size].unsqueeze(1)
            current_batch_e = expanded_emotional[i : i + batch_size]  # 暂时用不到

            # ----------------------------------------------------------
            # prepare tokens by eit
            prompt_embeds, pooled_prompt_embeds = encode_prompts_sdxl(pipeline, current_batch_n, accelerator.device)
            with autocast():
                # ── Emotion injection
                prompt_embeds = emo_model(inputs_embeds=prompt_embeds, arousal=current_batch_a.to(inference_dtype), valence=current_batch_v.to(inference_dtype))[0]
            # ----------------------------------------------------------

            if i % config.num_generations == 0:
                input_latents = global_input_latents.repeat(batch_size, 1, 1, 1).clone()

            with torch.no_grad():
                with autocast():
                    images, _, latents, log_probs = sdxlpipeline_with_logprob(
                        pipeline, prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, num_inference_steps=config.sample.num_steps, guidance_scale=config.sample.guidance_scale, eta=config.sample.eta, output_type="pt", latents=input_latents
                    )
            rewards = []
            for j, image in enumerate(images):
                pil = Image.fromarray((image.cpu().to(torch.float32).numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((512, 512))  # 1024 原始大小
                image_path = os.path.join(config.images_same_dir, f"rank-{dist.get_rank()}-image-{i}-{j}.jpg")
                pil.save(image_path)
                # Calculate the HPS
                with torch.no_grad():
                    with torch.amp.autocast("cuda"):
                        # TODO: prompt using current_batch_e
                        va_score = reward_model(pil, {"arousal": current_batch_a[j], "valence": current_batch_v[j], "prompt": current_batch_n[j]})
                        rewards.append(va_score)

            latents = torch.stack(latents, dim=1).detach()  # (4, num_steps+1, ...)
            log_probs = torch.stack(log_probs, dim=1).detach()  # (4, num_steps, ...)
            rewards = torch.cat(rewards, dim=0)

            all_latents.append(latents)
            all_log_probs.append(log_probs)
            all_rewards.append(rewards)
            all_prompts_embed.append(prompt_embeds)
            all_pooled_prompts_embed.append(pooled_prompt_embeds)  # ADD

            torch.cuda.empty_cache()

        all_latents = torch.cat(all_latents, dim=0)
        all_log_probs = torch.cat(all_log_probs, dim=0)
        all_rewards = torch.cat(all_rewards, dim=0).to(torch.float32)
        all_prompts_embed = torch.cat(all_prompts_embed, dim=0)
        all_pooled_prompts_embed = torch.cat(all_pooled_prompts_embed, dim=0)
        timesteps = pipeline.scheduler.timesteps.repeat(config.sample.batch_size * config.num_generations, 1)

        # time.sleep(0)
        # print("****************> all_latents shape {}".format(all_latents.shape))
        samples = {
            "prompt_embeds": all_prompts_embed,
            "pooled_prompt_embeds": all_pooled_prompts_embed,
            "timesteps": timesteps[:, :-1],
            "latents": all_latents[:, :-1],  # each entry is the latent before timestep t
            "next_latents": all_latents[:, 1:],  # each entry is the latent after timestep t
            "log_probs": all_log_probs[:, :-1],
            "rewards": all_rewards,
        }

        # gather rewards across processes
        all_rewards_world = gather_tensor(all_rewards)

        # log rewards and images
        accelerator.log({"reward": all_rewards_world, "epoch": epoch, "reward_mean": all_rewards_world.mean(), "reward_std": all_rewards_world.std()}, step=global_step)

        if dist.get_rank() == 0:
            os.makedirs(os.path.join(config.logdir, config.run_name), exist_ok=True)
            with open(os.path.join(config.logdir, config.run_name, "rewards.txt"), "a") as f:  # 'a'模式表示追加到文件末尾
                # with open(config.gathered_reward_path, "a") as f:  # 'a'模式表示追加到文件末尾
                f.write(f"{all_rewards_world.mean().item()}\n")

        # samples = {k: v.cuda() for k, v in samples.items()}  # 假设原始数据在GPU
        # samples = process_samples(samples, config)
        n = len(samples["rewards"]) // (config.num_generations)
        advantages = torch.zeros_like(samples["rewards"])

        for i in range(n):
            start_idx = i * config.num_generations
            end_idx = (i + 1) * config.num_generations
            group_rewards = samples["rewards"][start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
        samples["advantages"] = advantages

        samples["final_advantages"] = advantages

        total_batch_size, num_timesteps = samples["timesteps"].shape

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle along time dimension independently for each sample
            perms = torch.stack([torch.randperm(num_timesteps, device=accelerator.device) for _ in range(total_batch_size)])
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][torch.arange(total_batch_size, device=accelerator.device)[:, None], perms]

            # rebatch for training
            samples_batched = {k: v.reshape(-1, config.train.batch_size, *v.shape[1:]) for k, v in samples.items()}

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

            # train
            pipeline.unet.train()
            info = defaultdict(list)
            for i, sample in tqdm(list(enumerate(samples_batched)), desc=f"Epoch {epoch}.{inner_epoch}: training", position=0, disable=not accelerator.is_local_main_process):
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat([train_neg_prompt_embeds, sample["prompt_embeds"]])
                    pool_embeds = torch.cat([train_neg_pooled_embeds, sample["pooled_prompt_embeds"]])

                for j in tqdm(range(num_train_timesteps), desc="Timestep", position=1, leave=False, disable=not accelerator.is_local_main_process):
                    with accelerator.accumulate(unet):
                        with autocast():
                            if config.train.cfg:
                                latent_model_input = torch.cat([sample["latents"][:, j]] * 2)
                                _t = torch.cat([sample["timesteps"][:, j]] * 2)
                                # if accelerator.is_main_process:
                                # -------------
                                height = pipeline.default_sample_size * pipeline.vae_scale_factor
                                width = pipeline.default_sample_size * pipeline.vae_scale_factor
                                add_time_ids = pipeline._get_add_time_ids((height, width), (0, 0), (height, width), dtype=prompt_embeds.dtype, text_encoder_projection_dim=1280)  # pipeline.text_encoder_2.config.projection_dim,
                                add_time_ids = add_time_ids.repeat(latent_model_input.shape[0], 1).to(accelerator.device)
                                # -------------
                                added_cond_kwargs = {"text_embeds": pool_embeds, "time_ids": add_time_ids}

                                noise_pred = unet(latent_model_input, _t, embeds, added_cond_kwargs=added_cond_kwargs).sample  # prompt_embeds  #
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)
                            else:
                                # noise_pred = unet(sample["latents"][:, j], sample["timesteps"][:, j], embeds).sample
                                # not valid
                                pass

                            # compute the log prob of next_latents given latents under the current model
                            _, log_prob = ddim_step_with_logprob(pipeline.scheduler, noise_pred, sample["timesteps"][:, j], sample["latents"][:, j], eta=config.sample.eta, prev_sample=sample["next_latents"][:, j])

                        # ppo logic
                        advantages = torch.clamp(sample["final_advantages"], -config.train.adv_clip_max, config.train.adv_clip_max)
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(ratio, 1.0 - config.train.clip_range, 1.0 + config.train.clip_range)
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))  # policy loss
                        # kl_loss

                        # debugging values
                        # John Schulman says that (ratio - 1) - log(ratio) is a better
                        # estimator, but most existing code uses this so...
                        # http://joschu.net/blog/kl-approx.html
                        info["approx_kl"].append(0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2))
                        info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float()))
                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        assert (j == num_train_timesteps - 1) and (i + 1) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if dist.get_rank() % 8 == 0:
                    print("reward", sample["rewards"])
                    print("ratio", ratio)
                    print("final advantage", advantages)
                    print("hps_advantage", sample["advantages"])
                    print("final loss", loss)
                dist.barrier()

            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients
        # torch.cuda.empty_cache()
        if epoch != 0 and epoch % config.save_freq == 0:  #
            base_checkpoint_dir = config.my_checkpoints_dir
            checkpoint_epoch_dir = os.path.join(base_checkpoint_dir, f"checkpoint_epoch_{epoch}")
            if accelerator.is_main_process:
                # Create a unique directory for this specific checkpoint
                os.makedirs(checkpoint_epoch_dir, exist_ok=True)

                # Define paths for the UNet weights (safetensors) and config (json)
                unwrapped_unet = accelerator.unwrap_model(pipeline.unet)

                # 1. Save the UNet state_dict using safetensors
                try:
                    # The state_dict should contain only the model weights
                    state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
                    pipeline.save_lora_weights(save_directory=checkpoint_epoch_dir, unet_lora_layers=state_dict)
                    accelerator.print(f"Manually saved UNet weights to {checkpoint_epoch_dir}")
                except Exception as e:
                    accelerator.print(f"Error saving UNet weights with safetensors: {e}")

            # Barrier for distributed training, if initialized
            # Ensures all processes wait until the main process has finished saving
            if dist.is_initialized():
                dist.barrier()


if __name__ == "__main__":
    app.run(main)
