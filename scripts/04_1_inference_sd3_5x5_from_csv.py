# scripts/scripts/04_1_inference_sd3_5x5_from_csv.py
import csv
import os
import types
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ===== CONFIGURATION =====
USE_EMO_PROMPT = True  # 是否使用情感提示词
MODEL_TYPE = "flux"  # 可选: "flux", "omnigen2", "sd3"

# 模型路径配置
FLUX_PATH = "black-forest-labs/FLUX.1-dev"  # FLUX.1-dev 模型路径
OMNIGEN2_PATH = "OmniGen2/OmniGen2"  # OmniGen2 模型路径
SD3_PATH = "/WEIGHTS/WZX/tiny-sd3-pipe/"  # 原SD3模型路径（备用）
EIT_CKPT = ""  # EIT权重路径
LORA_PATH = ""

# 推理参数配置
SEED = 42
HEIGHT = 1024  # FLUX和OmniGen2默认1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 50  # FLUX默认50，OmniGen2默认50
GUIDANCE_SCALE = 3.5  # FLUX默认3.5
text_guidance_scale = 4.0  # OmniGen2专用
image_guidance_scale = 2.0  # OmniGen2专用（图生图时用到）

# 输入输出路径
CSV_PATH = "data/test_grpo_gpt4.csv"
OUTPUT_DIR = "/logs/wzx_data/results/5x5_from_csv"

# 显存优化配置（按需启用）
ENABLE_CPU_OFFLOAD = False  # 启用模型CPU卸载 ...最低40G显存
ENABLE_TAYLORSEER = False  # OmniGen2专用加速
ENABLE_TEACACHE = False  # OmniGen2专用加速（与TAYLORSEER互斥）
# =========================
# -----------------------------------------------------------------------------
# default config values
# -----------------------------------------------------------------------------
# make sure all variant in this file!
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and not k in ["sys", "os", "pd", "ET", "defaultdict", "Path", "random", "literal_eval"] and not isinstance(v, (types.ModuleType, type, types.FunctionType, types.BuiltinFunctionType)) and not callable(v) and k.isupper()  # 排除导入的模块/常用别名
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

if os.path.exists(EIT_CKPT) and USE_EMO_PROMPT:
    raise ValueError("EIT_CKPT and USE_EMO_PROMPT cannot be enabled simultaneously")

# Check dependency between EIT_CKPT and LORA_PATH
if os.path.exists(EIT_CKPT) and not os.path.exists(LORA_PATH):
    raise ValueError(f"When EIT_CKPT exists, LORA_PATH must also exist. Current LORA_PATH='{LORA_PATH}' does not exist")


# 导入分布式通信模块
try:
    from eit_with_anchor_and_grpo.utils.dist_comm import (enable_distributed,
                                                          get_global_size,
                                                          get_local_rank)

    DIST_AVAILABLE = True
except ImportError:
    DIST_AVAILABLE = False
    print("警告: 分布式模块不可用，将使用单卡模式")

from eit_with_anchor_and_grpo.models.rewards import EmotiCrafterEvaluator


def setup_distributed():
    """设置分布式环境"""
    use_torchrun = "RANK" in os.environ

    if use_torchrun and DIST_AVAILABLE:
        enable_distributed(use_torchrun=True)
        rank = get_local_rank()
        world_size = get_global_size()
        # 确保设备号不超过可用GPU数量
        device_id = rank % torch.cuda.device_count()
        device = f"cuda:{device_id}"
        print(f"[Rank {rank}/{world_size}] 分布式模式已启用，使用设备: {device}")
    else:
        rank = 0
        world_size = 1
        device = "cuda:0"
        print("单卡模式")

    return rank, world_size, device


def split_data_by_rank(data, rank, world_size):
    """按rank分割数据"""
    total = len(data)
    per_rank = (total + world_size - 1) // world_size
    start = rank * per_rank
    end = min(start + per_rank, total)
    return data[start:end], start, end


def load_pipeline(model_type, device):
    """加载指定的模型管道"""

    if model_type == "flux":
        from diffusers import FluxPipeline

        print(f"[Rank {get_local_rank() if DIST_AVAILABLE else 0}] 加载 FLUX.1-dev 模型...")
        pipeline = FluxPipeline.from_pretrained(FLUX_PATH, torch_dtype=torch.float16)

        if ENABLE_CPU_OFFLOAD:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline = pipeline.to(device)

        return pipeline, None, None, None

    elif model_type == "omnigen2":
        from accelerate import Accelerator

        from omnigen2.models.transformers.transformer_omnigen2 import \
            OmniGen2Transformer2DModel
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import \
            OmniGen2Pipeline

        print(f"[Rank {get_local_rank() if DIST_AVAILABLE else 0}] 加载 OmniGen2 模型...")

        # 正确初始化Accelerator
        accelerator = Accelerator(mixed_precision="fp16", device_placement=True)

        # 确保使用正确的设备
        if DIST_AVAILABLE:
            device = accelerator.device

        weight_dtype = torch.float16

        pipeline = OmniGen2Pipeline.from_pretrained(OMNIGEN2_PATH, torch_dtype=weight_dtype, trust_remote_code=True)

        # 加载transformer
        pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(OMNIGEN2_PATH, subfolder="transformer", torch_dtype=weight_dtype)

        # 启用加速选项
        if ENABLE_TAYLORSEER:
            pipeline.enable_taylorseer = True
        elif ENABLE_TEACACHE:
            pipeline.transformer.enable_teacache = True
            pipeline.transformer.teacache_rel_l1_thresh = 0.05

        # 显存优化
        if ENABLE_CPU_OFFLOAD:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline = pipeline.to(device)

        return pipeline, accelerator, None, None

    elif model_type == "sd3":
        # 保留原SD3逻辑
        from diffusers import StableDiffusion3Pipeline
        from transformers import GPT2Config

        print(f"[Rank {get_local_rank() if DIST_AVAILABLE else 0}] 加载 SD3 模型...")
        pipeline = StableDiffusion3Pipeline.from_pretrained(SD3_PATH, torch_dtype=torch.float16)
        pipeline.to(device)
        if os.path.exists(LORA_PATH):
            from peft import LoraConfig, PeftModel, get_peft_model

            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, LORA_PATH)
            # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
            pipeline.transformer.set_adapter("default")
            print("lora init ....")

        # 初始化EIT
        if not USE_EMO_PROMPT:
            from safetensors.torch import load_file

            from eit_with_anchor_and_grpo.models.eit import \
                EmotionInjectionTransformer

            config = GPT2Config.from_pretrained("./configs")
            sd_feature_dim = 32 if "tiny" in  LORA_PATH else 4096
            eit = EmotionInjectionTransformer(config, final_out_type="Linear+LN", sd_feature_dim=sd_feature_dim).to(device)
            if EIT_CKPT:
                state = load_file(EIT_CKPT)

                # Remove 'module.' prefix if present
                new_state_dict = {}
                for k, v in state.items():
                    if k.startswith("module."):
                        k = k[7:]  # Remove 'module.'
                    new_state_dict[k] = v
                eit.load_state_dict(new_state_dict)
                print(f"[EIT] Loaded checkpoint: {EIT_CKPT}")

            eit.eval()
        else:
            eit = None

        text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
        tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

        return pipeline, None, eit, (text_encoders, tokenizers)

    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def generate_with_flux(pipeline, prompt, emotional_prompt, device="cuda", seed=42):
    """使用FLUX生成图像"""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    with torch.no_grad():
        image = pipeline(prompt=emotional_prompt, height=HEIGHT, width=WIDTH, guidance_scale=GUIDANCE_SCALE, num_inference_steps=NUM_INFERENCE_STEPS, max_sequence_length=512, generator=generator).images[0]

    return image


def generate_with_omnigen2(pipeline, accelerator, prompt, emotional_prompt, device="cuda", seed=42):
    """使用OmniGen2生成图像"""
    from PIL import ImageOps

    # 使用accelerator的设备
    generator = torch.Generator(device=accelerator.device).manual_seed(seed)

    # 负向提示词
    negative_prompt = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"

    with torch.no_grad():
        results = pipeline(
            prompt=emotional_prompt,
            input_images=None,  # 纯文生图
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            max_sequence_length=1024,
            text_guidance_scale=text_guidance_scale,
            image_guidance_scale=image_guidance_scale,
            cfg_range=(0.0, 1.0),
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            generator=generator,
            output_type="pil",
        )

    return results.images[0]


def generate_with_sd3(pipeline, eit, text_encoders, tokenizers, prompt, v, a, device="cuda", seed=42):
    """使用SD3生成图像 - 支持EIT模式和原始pipeline模式

    当 eit 为 None 时，使用原始 pipeline 直接生成（情感提示词模式）
    当 eit 不为 None 时，使用 EIT 进行情感注入生成
    """
    from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob_fast import \
        pipeline_with_logprob
    from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import \
        encode_prompt

    with torch.no_grad():
        neg_embed, neg_pooled = encode_prompt(text_encoders, tokenizers, [""], max_sequence_length=128)
        pos_embed, pos_pooled = encode_prompt(text_encoders, tokenizers, [prompt], max_sequence_length=128)

        if eit is not None:
            pos_embed = eit(inputs_embeds=pos_embed.to(torch.float32), arousal=torch.FloatTensor([[a]]).to(device), valence=torch.FloatTensor([[v]]).to(device))[0]

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
            images, _, _, _ = pipeline_with_logprob(
                pipeline, prompt_embeds=pos_embed, pooled_prompt_embeds=pos_pooled, negative_prompt_embeds=neg_embed, negative_pooled_prompt_embeds=neg_pooled, num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE, output_type="pil", height=HEIGHT, width=WIDTH, generator=generator
            )

    return images[0]


def generate_with_emo_prompt(pipeline, model_type, accelerator, prompt, emotional_prompt, device="cuda", seed=42):
    """统一入口：使用情感提示词生成"""
    if model_type == "flux":
        return generate_with_flux(pipeline, prompt, emotional_prompt, device, seed)
    elif model_type == "omnigen2":
        return generate_with_omnigen2(pipeline, accelerator, prompt, emotional_prompt, device, seed)
    else:
        raise ValueError(f"模型 {model_type} 不支持情感提示词模式")


def create_5x5_for_prompt(prompt, group_df, pipeline, model_type, accelerator=None, eit=None, text_encoders=None, tokenizers=None, evaluator=None, device="cuda", seed=42):
    """为单个提示词生成5x5网格"""
    if len(group_df) != 25:
        print(f"警告: {prompt} 只有 {len(group_df)} 个样本，需要25个")
        return None

    # 排序：Arousal从-3到3，Valence从-3到3
    group_df = group_df.sort_values(["Arousal", "Valence"], ascending=[False, True])

    images = []
    metrics_list = []

    # 生成所有图像
    for _, row in group_df.iterrows():
        v, a = row["Valence"], row["Arousal"]

        if USE_EMO_PROMPT:
            emotional_prompt = row["Emotional_Prompt"]
            if model_type in ["flux", "omnigen2"]:
                image = generate_with_emo_prompt(pipeline, model_type, accelerator, prompt, emotional_prompt, device, seed)
            else:
                image = generate_with_sd3(pipeline, eit, text_encoders, tokenizers, prompt, v, a, device, seed)
        else:
            if model_type == "sd3":
                image = generate_with_sd3(pipeline, eit, text_encoders, tokenizers, prompt, v, a, device, seed)
            else:
                raise ValueError(f"模型 {model_type} 不支持EIT模式，请设置 USE_EMO_PROMPT=True")

        images.append(image)

        # 计算指标
        metrics = evaluator.evaluate(image, v, a, prompt)
        metrics_list.append(metrics)

    return images, metrics_list, group_df


def save_5x5_grid(images, group_df, prompt, metrics_list=None, rank=0):
    """保存5x5网格图"""
    fig = plt.figure(figsize=(20, 20))
    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    for idx, (image, (_, row)) in enumerate(zip(images, group_df.iterrows()), 1):
        plt.subplot(5, 5, idx)
        plt.axis("off")
        plt.imshow(image)

        # 添加目标VA值
        v, a = row["Valence"], row["Arousal"]
        plt.text(0.02, 0.98, f"V:{v}, A:{a}", transform=plt.gca().transAxes, fontsize=8, color="white", verticalalignment="top", bbox=dict(boxstyle="round", facecolor="black", alpha=0.5))

        # 添加预测值（如果有）
        if metrics_list:
            metrics = metrics_list[idx - 1]
            plt.text(0.02, 0.02, f"V':{metrics['pred_v']:.1f}, A':{metrics['pred_a']:.1f}", transform=plt.gca().transAxes, fontsize=8, color="white", verticalalignment="bottom", bbox=dict(boxstyle="round", facecolor="black", alpha=0.5))

    # 构建标题
    title = f"5x5 Grid ({MODEL_TYPE.upper()}): {prompt}"
    if metrics_list:
        v_errors = [m["v_error"] for m in metrics_list]
        a_errors = [m["a_error"] for m in metrics_list]
        title += f"\nV-Error: {np.mean(v_errors):.3f} ± {np.std(v_errors):.3f}, "
        title += f"A-Error: {np.mean(a_errors):.3f} ± {np.std(a_errors):.3f}"

    plt.suptitle(title, fontsize=12, y=0.95)

    # 保存文件（添加rank标识）
    prompt_safe = prompt.replace(" ", "_").replace("/", "_")[:50]
    save_path = Path(OUTPUT_DIR) / f"rank{rank}_5x5_{MODEL_TYPE}_{prompt_safe}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[Rank {rank}] 网格图已保存: {save_path}")

    return save_path


def save_metrics_csv(metrics_list, group_df, prompt, rank=0):
    """保存指标到CSV"""
    prompt_safe = prompt.replace(" ", "_").replace("/", "_")[:50]
    save_path = Path(OUTPUT_DIR) / f"rank{rank}_metrics_{MODEL_TYPE}_{prompt_safe}.csv"

    # 检查是否有第二个模型的指标
    has_second_model = "clip_score2" in metrics_list[0] if metrics_list else False

    # 定义字段名
    fieldnames = ["target_v", "target_a", "pred_v", "pred_a", "v_error", "a_error", "clip_score", "clip_iqa"]

    if has_second_model:
        fieldnames.extend(["clip_score2", "clip_iqa2"])

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for (_, row), metrics in zip(group_df.iterrows(), metrics_list):
            row_data = {"target_v": row["Valence"], "target_a": row["Arousal"], "pred_v": metrics["pred_v"], "pred_a": metrics["pred_a"], "v_error": metrics["v_error"], "a_error": metrics["a_error"], "clip_score": metrics["clip_score"], "clip_iqa": metrics["clip_iqa"]}

            if has_second_model:
                row_data.update({"clip_score2": metrics.get("clip_score2", 0), "clip_iqa2": metrics.get("clip_iqa2", 0)})

            writer.writerow(row_data)

    print(f"[Rank {rank}] 指标已保存: {save_path}")
    return save_path


def main():
    """主函数"""
    # 设置分布式
    rank, world_size, device = setup_distributed()

    # 添加分布式同步
    if DIST_AVAILABLE:
        torch.distributed.barrier()

    # 读取CSV数据
    print(f"[Rank {rank}] 读取CSV文件: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # 确保所有rank都读取完数据
    if DIST_AVAILABLE:
        torch.distributed.barrier()

    # 检查必要列
    required_cols = ["Neutral_Prompt", "Valence", "Arousal"]
    if USE_EMO_PROMPT:
        required_cols.append("Emotional_Prompt")

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV缺少必要列: {col}")

    # 按提示词分组
    grouped = df.groupby("Neutral_Prompt")
    prompt_groups = [(prompt, group) for prompt, group in grouped]

    # 按rank分割提示词组
    my_prompts, start_idx, end_idx = split_data_by_rank(prompt_groups, rank, world_size)
    print(f"[Rank {rank}] 负责提示词区间 [{start_idx}, {end_idx})，共 {len(my_prompts)} 个提示词")

    # 加载模型管道
    pipeline, accelerator, eit, encoders_tokenizers = load_pipeline(MODEL_TYPE, device)

    if encoders_tokenizers:
        text_encoders, tokenizers = encoders_tokenizers
    else:
        text_encoders = tokenizers = None

    # 禁用进度条（分布式环境下）
    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=True)

    # 初始化评估器
    print(f"[Rank {rank}] 初始化evaluator...")
    evaluator = EmotiCrafterEvaluator(
        arousal_ckpt="metrics/arousal1_CLIP_lr=0.001_loss=MSELoss_sc=test_cuda-1.pth",
        valence_ckpt="metrics/valence1_CLIP_lr=0.0001_loss=MSELoss_sc=test_cuda.pth",
        # model_name_or_path="/WEIGHTS/WZX/openai/clip-vit-large-patch14",
        model_name_or_path="/WEIGHTS/WZX/openai/clip-vit-base-patch32",
        # model_name_or_path_2="/WEIGHTS/WZX/openai/clip-vit-base-patch32",
        device=device,
    )

    # 处理分配到的提示词
    all_results = []

    for prompt_idx, (prompt, group_df) in enumerate(my_prompts, 1):
        print(f"[Rank {rank}] [{prompt_idx}/{len(my_prompts)}] 处理提示词: {prompt}")

        # 生成5x5网格
        result = create_5x5_for_prompt(prompt, group_df, pipeline, MODEL_TYPE, accelerator, eit, text_encoders, tokenizers, evaluator, device, SEED)

        if result is None:
            continue

        images, metrics_list, sorted_df = result

        # 保存结果
        grid_path = save_5x5_grid(images, sorted_df, prompt, metrics_list, rank)

        if metrics_list:
            metrics_path = save_metrics_csv(metrics_list, sorted_df, prompt, rank)
        else:
            metrics_path = None

        all_results.append(
            {"prompt": prompt, "grid_path": str(grid_path), "metrics_path": str(metrics_path) if metrics_path else None, "v_error_mean": np.mean([m["v_error"] for m in metrics_list]) if metrics_list else None, "a_error_mean": np.mean([m["a_error"] for m in metrics_list]) if metrics_list else None}
        )

    # 保存本rank的汇总结果
    if all_results:
        summary_path = Path(OUTPUT_DIR) / f"rank{rank}_summary_{MODEL_TYPE}.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["prompt", "grid_path", "metrics_path", "v_error_mean", "a_error_mean"])
            writer.writeheader()
            writer.writerows(all_results)

        print(f"[Rank {rank}] 汇总结果已保存: {summary_path}")

    print(f"[Rank {rank}] 处理完成！共处理 {len(all_results)} 个提示词")


if __name__ == "__main__":
    main()
# torchrun --nproc_per_node=4 scripts/04_1_inference_sd3_5x5_from_csv.py
