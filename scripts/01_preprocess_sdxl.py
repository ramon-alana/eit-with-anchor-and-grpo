"""
python scripts/01_preprocess_sdxl.py
torchrun --master_port=40001 --nnodes=1 --nproc_per_node=8
"""

import csv
import logging
import os

import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                          CLIPTokenizer)

from eit_with_anchor_and_grpo.utils.dist_comm import (enable_distributed,
                                                      get_global_rank,
                                                      get_global_size,
                                                      get_local_rank)
from eit_with_anchor_and_grpo.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# ============== Configuration ==============
TEXT_ENCODER_PATH = "/WEIGHTS/WZX/stable-diffusion-xl-base-1.0/"  # "/data0/data/wzx_data/emov2"  # SDXL path, but only load text encoders "/WEIGHTS/WZX/stable-diffusion-xl-base-1.0/"
CSV_PATH = "./data/prompt_mapping.csv"
DATA_CACHE_PATH = "./data/data-sdxl-fp16.safetensors"
BATCH_SIZE = 1  # Adjust based on your GPU memory
# ===========================================
DEVICE = "cuda"

use_torchrun = os.environ.get("LOCAL_RANK") is not None


# ------------
#     dist sampler get
# ------------
def cal_rank_task_num(total_num, rank, world_size, batch_size):
    """计算当前rank需要处理的batch数量"""
    total_batches = (total_num + batch_size - 1) // batch_size  # 向上取整
    base_num = total_batches // world_size
    remainder = total_batches % world_size
    return base_num + 1 if rank < remainder else base_num


def get_rank_specific_batches(data, rank, world_size, batch_size, skip_num=0):
    """按batch分配任务给不同rank"""
    # 先按batch_size分组
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i : i + batch_size])

    total_batches = len(batches)
    match_counts = 0
    for idx, batch in enumerate(batches):
        if idx % world_size == rank:
            match_counts += 1
            if match_counts <= skip_num:
                continue
            yield (match_counts, idx, batch)


def load_pipeline_method(path, device="cuda", dtype=torch.float16):
    """
    preprocess.py 的方法: 使用 StableDiffusionXLPipeline.encode_prompt
    """
    pipe = StableDiffusionXLPipeline.from_pretrained(path, torch_dtype=dtype, use_safetensors=True, variant="fp16")
    pipe.to(device)
    return pipe


def load_text_encoders(path, device="cuda", dtype=torch.float16):
    """Load only SDXL text encoders (CLIP-L + OpenCLIP-G), skip UNet/VAE to save VRAM"""
    # Text Encoder 1 (CLIP-L, 768-dim)
    tokenizer_1 = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer")
    text_encoder_1 = CLIPTextModel.from_pretrained(path, subfolder="text_encoder", torch_dtype=dtype).to(device)

    # Text Encoder 2 (OpenCLIP-G, 1280-dim)
    tokenizer_2 = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(path, subfolder="text_encoder_2", torch_dtype=dtype).to(device)

    # Freeze and eval mode
    text_encoder_1.eval()
    text_encoder_2.eval()
    for param in text_encoder_1.parameters():
        param.requires_grad = False
    for param in text_encoder_2.parameters():
        param.requires_grad = False

    return tokenizer_1, text_encoder_1, tokenizer_2, text_encoder_2


def text_encode_prompts_batch(tokenizer_1, text_encoder_1, tokenizer_2, text_encoder_2, prompts, device, dtype=torch.float16):
    # Text Encoder 1 (CLIP-L)
    ids_1 = tokenizer_1(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=tokenizer_1.model_max_length).input_ids.to(device)
    hidden_1 = text_encoder_1(ids_1, output_hidden_states=True).hidden_states[-2]  # [B, 77, 768]

    # Text Encoder 2 (OpenCLIP-G)
    ids_2 = tokenizer_2(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=tokenizer_2.model_max_length).input_ids.to(device)
    output_2 = text_encoder_2(ids_2, output_hidden_states=True)
    hidden_2 = output_2.hidden_states[-2]  # [B, 77, 1280]
    pooled = output_2[0]  # [B, 1280]

    # Concatenate: [B, 77, 2048]
    embeds = torch.cat([hidden_1, hidden_2], dim=-1)
    return embeds, pooled


def batched(iterable, n):
    """Batch iterator"""
    from itertools import islice

    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


def encode_with_pipeline(pipe, prompts, device):
    """
    preprocess.py 的 encode 方式
    返回: prompt_embeds [B, 77, 2048], pooled_prompt_embeds [B, 1280]
    """
    with torch.no_grad():
        (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
            prompt=prompts, prompt_2=prompts, device=device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=None, negative_prompt_2=None, prompt_embeds=None, negative_prompt_embeds=None, pooled_prompt_embeds=None, negative_pooled_prompt_embeds=None
        )
    # logger.info(prompt_embeds.shape)
    prompt_embeds = prompt_embeds  # [B, 77, 2048]
    pooled_prompt_embeds = pooled_prompt_embeds  # [B, 1280]

    return prompt_embeds, pooled_prompt_embeds


if __name__ == "__main__":

    setup_logging(level=logging.INFO)
    if use_torchrun:
        enable_distributed(use_torchrun=use_torchrun)
    # 获取本地rank并设置设备
    try:
        rank = get_local_rank()
    except:  # noqa: E722
        rank = get_global_rank()
    world_size = get_global_size()
    # Load data
    with open(CSV_PATH, "r") as f:
        data = list(csv.DictReader(f)) # Remove [:10] for full dataset

    assert len(data) >= world_size

    device = f"cuda:{rank}"
    run_task_num = cal_rank_task_num(total_num=len(data), rank=rank, world_size=world_size, batch_size=BATCH_SIZE)
    skip_num = 0
    pbar = tqdm(get_rank_specific_batches(data, rank, world_size, batch_size=BATCH_SIZE, skip_num=skip_num), total=run_task_num, disable=rank != 0)
    logger.info(f"total feature numbers:{len(data)}| current process numbers: {run_task_num}| bs:{BATCH_SIZE} |skips: {skip_num}|world_size: {world_size}|rank:{rank} ")

    # Load text encoders only (~2GB VRAM vs ~8GB for full SDXL)
    logger.info("Loading text encoders...")

    pipeline = load_text_encoders(TEXT_ENCODER_PATH, device=device)
    encode_prompts_fn = text_encode_prompts_batch

    # Collect results
    all_neutral_embeds, all_neutral_pooled = [], []
    all_emotional_embeds, all_emotional_pooled = [], []
    all_arousal, all_valence = [], []
    all_neutral_prompts, all_emotional_prompts = [], []  # Keep raw prompts if needed

    logger.info(f"Processing {len(data)} samples with batch_size={BATCH_SIZE}...")
    with torch.no_grad():
        # for batch in tqdm(batched(data, BATCH_SIZE), total=(len(data) + BATCH_SIZE - 1) // BATCH_SIZE):
        for skipped_id, global_idx, batch in pbar:
            # Extract batch data
            neutral_prompts = [item["Neutral_Prompt"] for item in batch]
            emotional_prompts = [item["Emotional_Prompt"] for item in batch]
            arousals = [float(item["Arousal"]) for item in batch]
            valences = [float(item["Valence"]) for item in batch]

            # Batch encode
            n_embeds, n_pooled = encode_prompts_fn(*pipeline, neutral_prompts, "cuda")
            e_embeds, e_pooled = encode_prompts_fn(*pipeline, emotional_prompts, "cuda")

            # Store (keep on GPU until end, or .cpu() immediately to save VRAM)
            all_neutral_embeds.append(n_embeds.cpu())
            all_neutral_pooled.append(n_pooled.cpu())
            all_emotional_embeds.append(e_embeds.cpu())
            all_emotional_pooled.append(e_pooled.cpu())
            all_arousal.extend(arousals)
            all_valence.extend(valences)
            all_neutral_prompts.extend(neutral_prompts)
            all_emotional_prompts.extend(emotional_prompts)

    # 每个rank保存自己的shard
    shard_path = DATA_CACHE_PATH.replace(".safetensors", f".rank{rank}.safetensors")
    shard_prompt_path = DATA_CACHE_PATH.replace(".safetensors", f".rank{rank}.prompt.txt")

    local_data_dict = {
        "neutral_prompt_feature": torch.cat(all_neutral_embeds).to(torch.float16),
        "neutral_pooled_feature": torch.cat(all_neutral_pooled).to(torch.float16),
        "emotional_prompt_feature": torch.cat(all_emotional_embeds).to(torch.float16),
        "emotional_pooled_feature": torch.cat(all_emotional_pooled).to(torch.float16),
        "arousal": torch.tensor(all_arousal, dtype=torch.float).unsqueeze(1),
        "valence": torch.tensor(all_valence, dtype=torch.float).unsqueeze(1),
    }
    save_file(local_data_dict, shard_path)
    # Replace: np.savetxt(shard_prompt_path, list(zip(all_neutral_prompts, all_emotional_prompts)), fmt="%s", delimiter="\t")

    with open(shard_prompt_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for n, e in zip(all_neutral_prompts, all_emotional_prompts):
            writer.writerow([n, e])
    logger.info(f"Rank {rank}: Saved shard to {shard_path}")

    # Barrier等待所有rank完成
    if world_size > 1:
        torch.distributed.barrier()

    # Rank 0合并所有shard
    if rank == 0:
        logger.info("Rank 0: Merging shards...")
        all_data = {"neutral_prompt_feature": [], "neutral_pooled_feature": [], "emotional_prompt_feature": [], "emotional_pooled_feature": [], "arousal": [], "valence": []}
        all_neutral_prompts, all_emotional_prompts = [], []

        for r in range(world_size):
            shard_path_r = DATA_CACHE_PATH.replace(".safetensors", f".rank{r}.safetensors")
            shard = load_file(shard_path_r)  # from safetensors import load_file

            for key in all_data.keys():
                all_data[key].append(shard[key])

            # 读取prompt文件
            prompts_r = []
            with open(DATA_CACHE_PATH.replace(".safetensors", f".rank{r}.prompt.txt"), "r") as f:
                reader = csv.reader(f, delimiter="\t")
                prompts_r = list(reader)
            prompts_r = np.array(prompts_r)
            all_neutral_prompts.extend(prompts_r[:, 0].tolist())
            all_emotional_prompts.extend(prompts_r[:, 1].tolist())

            # 删除shard
            os.remove(shard_path_r)
            os.remove(DATA_CACHE_PATH.replace(".safetensors", f".rank{r}.prompt.txt"))

        # 保存合并后的文件
        merged_data = {k: torch.cat(v) for k, v in all_data.items()}
        save_file(merged_data, DATA_CACHE_PATH)
        with open(DATA_CACHE_PATH.replace(".safetensors", ".prompt.txt"), "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            for n, e in zip(all_neutral_prompts, all_emotional_prompts):
                writer.writerow([n, e])
        logger.info(f"Rank 0: Saved merged file to {DATA_CACHE_PATH}")

    # 确保rank0完成后再退出
    if world_size > 1:
        torch.distributed.barrier()
