# summarize_metrics.py
"""
汇总多个CSV文件的评估指标统计信息
保持与 print_summary_statistics 相同的指标格式

用法:
python scripts/05_summarize_metrics.py --csv_dir "/logs/wzx_data/results/5x5_from_csv"  --output_dir "/logs/wzx_data/results/5x5_from_csv"

参数:
--csv_dir: 包含CSV文件的目录路径
--output_dir: 输出汇总结果的目录路径
--file_pattern: CSV文件匹配模式 (默认: "*metric*.csv")
"""

import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_all_csv_files(csv_dir, file_pattern="*metric*.csv"):
    """加载目录中的所有CSV文件"""
    csv_files = glob.glob(os.path.join(csv_dir, file_pattern))

    if not csv_files:
        raise ValueError(f"在目录 {csv_dir} 中未找到匹配 {file_pattern} 的CSV文件")

    print(f"找到 {len(csv_files)} 个CSV文件")

    all_data = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # 从文件名提取prompt信息
            filename = os.path.basename(csv_file)
            prompt_id = filename.replace("5x5_metrics_", "").replace(".csv", "")

            # 添加文件信息
            df["file_name"] = filename
            df["prompt_id"] = prompt_id

            all_data.append(df)
            print(f"  ✓ {filename}: {len(df)} 个样本")

        except Exception as e:
            print(f"  ✗ 加载 {csv_file} 失败: {e}")

    if not all_data:
        raise ValueError("没有成功加载任何CSV文件")

    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)

    return combined_df


def calculate_summary_statistics(df):
    """计算汇总统计信息"""
    v_errors = df["v_error"].values
    a_errors = df["a_error"].values
    clip_scores = df["clip_score"].values
    clip_iqas = df["clip_iqa"].values

    summary = {
        "total_samples": len(df),
        "v_error_mean": np.mean(v_errors),
        "v_error_std": np.std(v_errors),
        "a_error_mean": np.mean(a_errors),
        "a_error_std": np.std(a_errors),
        "clip_score_mean": np.mean(clip_scores),
        "clip_score_std": np.std(clip_scores),
        "clip_iqa_mean": np.mean(clip_iqas),
        "clip_iqa_std": np.std(clip_iqas),
    }

    # 检查是否有第二个模型的指标
    if "clip_score2" in df.columns:
        clip_scores2 = df["clip_score2"].values
        clip_iqas2 = df["clip_iqa2"].values

        summary.update({"clip_score2_mean": np.mean(clip_scores2), "clip_score2_std": np.std(clip_scores2), "clip_iqa2_mean": np.mean(clip_iqas2), "clip_iqa2_std": np.std(clip_iqas2)})

    return summary


def print_summary_statistics(summary):
    """打印汇总统计信息"""
    print("\n" + "=" * 60)
    print("评估指标汇总统计:")
    print("=" * 60)
    print(f"总样本数: {summary['total_samples']}")
    print(f"V-Error: {summary['v_error_mean']:.3f} ± {summary['v_error_std']:.3f}")
    print(f"A-Error: {summary['a_error_mean']:.3f} ± {summary['a_error_std']:.3f}")
    print(f"CLIPScore: {summary['clip_score_mean']:.3f} ± {summary['clip_score_std']:.3f}")
    print(f"CLIP-IQA: {summary['clip_iqa_mean']:.3f} ± {summary['clip_iqa_std']:.3f}")

    # 打印第二个模型的指标（如果存在）
    if "clip_score2_mean" in summary:
        print(f"CLIPScore2: {summary['clip_score2_mean']:.3f} ± {summary['clip_score2_std']:.3f}")
        print(f"CLIP-IQA2: {summary['clip_iqa2_mean']:.3f} ± {summary['clip_iqa2_std']:.3f}")

    print("=" * 60)


def save_summary_to_csv(summary, output_path):
    """保存汇总结果到CSV文件"""
    with open(output_path, "w", newline="") as csvfile:
        # 定义字段名
        fieldnames = ["metric", "mean", "std", "total_samples"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({"metric": "V-Error", "mean": f"{summary['v_error_mean']:.3f}", "std": f"{summary['v_error_std']:.3f}", "total_samples": summary["total_samples"]})
        writer.writerow({"metric": "A-Error", "mean": f"{summary['a_error_mean']:.3f}", "std": f"{summary['a_error_std']:.3f}", "total_samples": summary["total_samples"]})
        writer.writerow({"metric": "CLIPScore", "mean": f"{summary['clip_score_mean']:.3f}", "std": f"{summary['clip_score_std']:.3f}", "total_samples": summary["total_samples"]})
        writer.writerow({"metric": "CLIP-IQA", "mean": f"{summary['clip_iqa_mean']:.3f}", "std": f"{summary['clip_iqa_std']:.3f}", "total_samples": summary["total_samples"]})

        # 保存第二个模型的指标（如果存在）
        if "clip_score2_mean" in summary:
            writer.writerow({"metric": "CLIPScore2", "mean": f"{summary['clip_score2_mean']:.3f}", "std": f"{summary['clip_score2_std']:.3f}", "total_samples": summary["total_samples"]})
            writer.writerow({"metric": "CLIP-IQA2", "mean": f"{summary['clip_iqa2_mean']:.3f}", "std": f"{summary['clip_iqa2_std']:.3f}", "total_samples": summary["total_samples"]})

    print(f"汇总结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="汇总多个CSV文件的评估指标统计信息")
    parser.add_argument("--csv_dir", type=str, nargs="+", required=True, help="一个或多个包含CSV文件的目录路径")
    parser.add_argument("--output_dir", type=str, default="./summary", help="输出汇总结果的目录路径")
    parser.add_argument("--file_pattern", type=str, default="*metric*.csv", help="CSV文件匹配模式")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_df = []

    for csv_dir in args.csv_dir:
        print(f"正在处理目录: {csv_dir}")
        df = load_all_csv_files(csv_dir, args.file_pattern)
        all_df.append(df)

    final_df = pd.concat(all_df, ignore_index=True)
    print(f"\n处理完成，共 {len(final_df)} 个样本")

    # 保存所有原始数据
    final_df.to_csv(output_dir / "summary_all.csv", index=False)

    # 计算并打印汇总统计
    summary = calculate_summary_statistics(final_df)
    print_summary_statistics(summary)

    # 保存汇总结果
    save_summary_to_csv(summary, output_dir / "summary.csv")


if __name__ == "__main__":
    main()
