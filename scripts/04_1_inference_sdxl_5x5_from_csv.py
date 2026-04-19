# scripts/04_1_inference_sdxl_5x5_from_csv.py
"""
SDXL版本的5x5网格生成脚本，从CSV文件读取数据
支持分布式推理和评估指标计算
"""

import csv
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ===== CONFIGURATION =====
USE_EMO_PROMPT = True  # 是否使用情感提示词（跳过EIT和LoRA）
SDXL_PATH = "/data0/data/wzx_data/emov2"  # SDXL模型路径
CSV_PATH = "data/test_grpo_gpt4.csv"  # 输入CSV文件
OUTPUT_DIR = "/logs/wzx_data/results/sdxl_5x5_from_csv"  # 输出目录
SEED = 42
height = 1024
width = 1024
num_inference_steps = 25
guidance_scale = 7.5
# =========================

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

if USE_EMO_PROMPT:
    from diffusers import StableDiffusionXLPipeline
else:
    from diffusers import StableDiffusionXLPipeline
    from transformers import GPT2Config

    from eit_with_anchor_and_grpo.models.eit import EmotionInjectionTransformer


def setup_distributed():
    """设置分布式环境"""
    use_torchrun = "RANK" in os.environ

    if use_torchrun and DIST_AVAILABLE:
        enable_distributed(use_torchrun=True)
        rank = get_local_rank()
        world_size = get_global_size()
        device = f"cuda:{rank}"
        print(f"[Rank {rank}/{world_size}] 分布式模式已启用")
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


def generate_with_emo_prompt(pipeline, prompt, emotional_prompt, device="cuda", seed=42):
    """使用情感提示词直接生成"""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    with torch.no_grad():
        image = pipeline(prompt=emotional_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=height, width=width, generator=generator).images[0]

    return image


def generate_with_eit(pipeline, eit, prompt, v, a, device="cuda", seed=42):
    """使用EIT生成（SDXL版本）"""
    # SDXL使用两个文本编码器
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # 编码提示词
    with torch.no_grad():
        # SDXL的encode_prompt返回两个编码器的输出
        prompt_embeds, pooled_prompt_embeds = pipeline.encode_prompt(prompt=[prompt], prompt_2=[prompt], device=device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=[""], negative_prompt_2=[""])  # SDXL需要两个提示词，通常相同

        # 情感注入
        # 注意：SDXL的prompt_embeds形状可能不同，需要调整
        # 假设EIT期望的输入形状为 [batch_size, seq_len, hidden_dim]
        prompt_embeds = prompt_embeds.to(torch.float32)
        arousal = torch.FloatTensor([[a]]).to(device)
        valence = torch.FloatTensor([[v]]).to(device)

        # 通过EIT处理
        modified_embeds = eit(inputs_embeds=prompt_embeds, arousal=arousal, valence=valence)[0]

        # 生成图像
        image = pipeline(prompt_embeds=modified_embeds.to(torch.float16), pooled_prompt_embeds=pooled_prompt_embeds, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, height=height, width=width, generator=generator).images[0]

    return image


def create_5x5_for_prompt(prompt, group_df, pipeline, eit=None, evaluator=None, device="cuda", seed=42):
    """为单个提示词生成5x5网格"""
    # 确保有25个样本（5x5）
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
            image = generate_with_emo_prompt(pipeline, prompt, emotional_prompt, device, seed)
        else:
            image = generate_with_eit(pipeline, eit, prompt, v, a, device, seed)

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
    title = f"SDXL 5x5 Grid: {prompt}"
    if metrics_list:
        v_errors = [m["v_error"] for m in metrics_list]
        a_errors = [m["a_error"] for m in metrics_list]
        title += f"\nV-Error: {np.mean(v_errors):.3f} ± {np.std(v_errors):.3f}, "
        title += f"A-Error: {np.mean(a_errors):.3f} ± {np.std(a_errors):.3f}"

    plt.suptitle(title, fontsize=12, y=0.95)

    # 保存文件（添加rank标识）
    prompt_safe = prompt.replace(" ", "_").replace("/", "_")[:50]
    save_path = Path(OUTPUT_DIR) / f"rank{rank}_5x5_{prompt_safe}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[Rank {rank}] 网格图已保存: {save_path}")

    return save_path


def save_metrics_csv(metrics_list, group_df, prompt, rank=0):
    """保存指标到CSV"""
    prompt_safe = prompt.replace(" ", "_").replace("/", "_")[:50]
    save_path = Path(OUTPUT_DIR) / f"rank{rank}_metrics_{prompt_safe}.csv"

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

    # 读取CSV数据
    print(f"[Rank {rank}] 读取CSV文件: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

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

    # 初始化SDXL管道
    print(f"[Rank {rank}] 初始化SDXL管道...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(SDXL_PATH, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    # 初始化评估器
    print(f"[Rank {rank}] 初始化evaluator...")
    evaluator = EmotiCrafterEvaluator(
        arousal_ckpt="metrics/arousal1_CLIP_lr=0.001_loss=MSELoss_sc=test_cuda-1.pth",
        valence_ckpt="metrics/valence1_CLIP_lr=0.0001_loss=MSELoss_sc=test_cuda.pth",
        model_name_or_path="/WEIGHTS/WZX/openai/clip-vit-large-patch14",
        model_name_or_path_2="/WEIGHTS/WZX/openai/clip-vit-base-patch32",
        device=device,
    )

    # 非情感提示词模式需要额外初始化
    if not USE_EMO_PROMPT:
        print(f"[Rank {rank}] 初始化EIT...")

        # 初始化EIT（SDXL版本）
        config = GPT2Config.from_pretrained("./config")
        # 注意：SDXL的hidden_dim可能与SD3不同，需要调整
        eit = EmotionInjectionTransformer(config, final_out_type="Linear+LN", sd_feature_dim=2048).to(device)  # SDXL的隐藏维度

        # 加载检查点（如果需要）
        # eit.load_state_dict(...)
        eit.eval()
    else:
        eit = None

    # 处理分配到的提示词
    all_results = []

    for prompt_idx, (prompt, group_df) in enumerate(my_prompts, 1):
        print(f"[Rank {rank}] [{prompt_idx}/{len(my_prompts)}] 处理提示词: {prompt}")

        # 生成5x5网格
        result = create_5x5_for_prompt(prompt, group_df, pipeline, eit, evaluator, device, SEED)

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
        summary_path = Path(OUTPUT_DIR) / f"rank{rank}_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["prompt", "grid_path", "metrics_path", "v_error_mean", "a_error_mean"])
            writer.writeheader()
            writer.writerows(all_results)

        print(f"[Rank {rank}] 汇总结果已保存: {summary_path}")

    print(f"[Rank {rank}] 处理完成！共处理 {len(all_results)} 个提示词")


if __name__ == "__main__":
    main()
