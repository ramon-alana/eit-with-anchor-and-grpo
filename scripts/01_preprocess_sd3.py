"""
python scripts/01_preprocess_sd3.py
torchrun --master_port=40001 --nnodes=1 --nproc_per_node=8
"""

import csv
import logging
import os
import types

import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
from safetensors.torch import load_file, save_file  # 添加导入
from tqdm import tqdm

from eit_with_anchor_and_grpo.models.encode_sd3_prompt_fn import encode_prompt
from eit_with_anchor_and_grpo.utils.dist_comm import (enable_distributed,
                                                      get_global_rank,
                                                      get_global_size,
                                                      get_local_rank)
from eit_with_anchor_and_grpo.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# ============== Configuration ==============
MODEL_PATH = "/WEIGHTS/WZX/tiny-sd3-pipe/"  # /WEIGHTS/WZX/sd3.5-medium /WEIGHTS/WZX/tiny-sd3-pipe
CSV_PATH = "./data/prompt_mapping.csv"
DATA_CACHE_PATH = "data/data-sd3-fp16.safetensors"
BATCH_SIZE = 1  # Adjust based on your GPU memory
DEBUG = False
# ===========================================

use_torchrun = os.environ.get("LOCAL_RANK") is not None

# -----------------------------------------------------------------------------
# default config values
# -----------------------------------------------------------------------------
# make sure all variant in this file!
config_keys = [
    k for k, v in globals().items() if not k.startswith("_") and not k in ["sys", "os", "pd", "ET", "defaultdict", "Path", "random", "literal_eval"] and not isinstance(v, (types.ModuleType, type, types.FunctionType, types.BuiltinFunctionType)) and not callable(v)  and k.isupper()  # 排除导入的模块/常用别名
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

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


# ------------
def load_pipeline_method(path, device="cuda", dtype=torch.float16):
    pipe = StableDiffusion3Pipeline.from_pretrained(path, torch_dtype=dtype)
    pipe.to(device)
    return pipe


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length)
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
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

    # Load data - 移除测试限制
    with open(CSV_PATH, "r") as f:
        if DEBUG:
            data = list(csv.DictReader(f))[:10]
        else:
            data = list(csv.DictReader(f))

    if len(data) < world_size:
        logger.warning(f"数据量({len(data)})小于进程数({world_size})，某些rank可能没有任务")

    run_task_num = cal_rank_task_num(total_num=len(data), rank=rank, world_size=world_size, batch_size=BATCH_SIZE)
    skip_num = 0

    # 确保文件路径包含.safetensors后缀
    if not DATA_CACHE_PATH.endswith(".safetensors"):
        DATA_CACHE_PATH = DATA_CACHE_PATH + ".safetensors"

    pbar = tqdm(get_rank_specific_batches(data, rank, world_size, batch_size=BATCH_SIZE, skip_num=skip_num), total=run_task_num, disable=rank != 0)
    logger.info(f"total feature numbers:{len(data)}| current process numbers: {run_task_num}| bs:{BATCH_SIZE} |skips: {skip_num}|world_size: {world_size}|rank:{rank} ")

    # Collect results
    all_neutral_embeds, all_neutral_pooled = [], []
    all_emotional_embeds, all_emotional_pooled = [], []
    all_arousal, all_valence = [], []
    all_neutral_prompts, all_emotional_prompts = [], []  # Keep raw prompts if needed
    logger.info("loading  model>>>>>>")

    device = f"cuda:{rank}"
    pipeline = load_pipeline_method(path=MODEL_PATH, device=device)
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)

    # 删除transformer释放显存
    del pipeline.transformer
    torch.cuda.empty_cache()

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    logger.info(f"Processing {len(data)} samples with batch_size={BATCH_SIZE}...")
    with torch.no_grad():
        for skipped_id, global_idx, batch in pbar:
            # Extract batch data
            neutral_prompts = [item["Neutral_Prompt"] for item in batch]
            emotional_prompts = [item["Emotional_Prompt"] for item in batch]
            arousals = [float(item["Arousal"]) for item in batch]
            valences = [float(item["Valence"]) for item in batch]

            n_embeds, n_pooled = compute_text_embeddings(neutral_prompts, text_encoders, tokenizers, max_sequence_length=128, device=device)
            e_embeds, e_pooled = compute_text_embeddings(emotional_prompts, text_encoders, tokenizers, max_sequence_length=128, device=device)

            # Store (keep on GPU until end, or .cpu() immediately to save VRAM)
            all_neutral_embeds.append(n_embeds.cpu())
            all_neutral_pooled.append(n_pooled.cpu())
            all_emotional_embeds.append(e_embeds.cpu())
            all_emotional_pooled.append(e_pooled.cpu())
            all_arousal.extend(arousals)
            all_valence.extend(valences)
            all_neutral_prompts.extend(neutral_prompts)
            all_emotional_prompts.extend(emotional_prompts)

    # ========== 保存每个rank的shard ==========
    # 确保有数据才保存
    if len(all_neutral_embeds) > 0:
        base_path = DATA_CACHE_PATH.replace(".safetensors", "")
        shard_path = f"{base_path}.rank{rank}.safetensors"
        shard_prompt_path = f"{base_path}.rank{rank}.prompt.txt"

        local_data_dict = {
            "neutral_prompt_feature": torch.cat(all_neutral_embeds).to(torch.float16),
            "neutral_pooled_feature": torch.cat(all_neutral_pooled).to(torch.float16),
            "emotional_prompt_feature": torch.cat(all_emotional_embeds).to(torch.float16),
            "emotional_pooled_feature": torch.cat(all_emotional_pooled).to(torch.float16),
            "arousal": torch.tensor(all_arousal, dtype=torch.float).unsqueeze(1),
            "valence": torch.tensor(all_valence, dtype=torch.float).unsqueeze(1),
        }
        save_file(local_data_dict, shard_path)

        with open(shard_prompt_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            for n, e in zip(all_neutral_prompts, all_emotional_prompts):
                writer.writerow([n, e])
        logger.info(f"Rank {rank}: Saved {len(all_neutral_prompts)} samples to {shard_path}")

        del all_neutral_embeds, all_neutral_pooled, all_emotional_embeds, all_emotional_pooled
        del all_arousal, all_valence, all_neutral_prompts, all_emotional_prompts
        del local_data_dict
        import gc

        gc.collect()
    else:
        logger.info(f"Rank {rank}: No data to save")

    # Barrier等待所有rank完成
    if world_size > 1:
        torch.distributed.barrier()

    # Rank 0合并所有shard
    if rank == 0:
        logger.info("Rank 0: Merging shards...")
        all_data = {"neutral_prompt_feature": [], "neutral_pooled_feature": [], "emotional_prompt_feature": [], "emotional_pooled_feature": [], "arousal": [], "valence": []}
        all_neutral_prompts, all_emotional_prompts = [], []

        for r in range(world_size):
            base_path = DATA_CACHE_PATH.replace(".safetensors", "")
            shard_path_r = f"{base_path}.rank{r}.safetensors"
            shard_prompt_path_r = f"{base_path}.rank{r}.prompt.txt"

            if not os.path.exists(shard_path_r):
                logger.warning(f"Shard for rank {r} not found, skipping")
                continue

            shard = load_file(shard_path_r)

            for key in all_data.keys():
                all_data[key].append(shard[key])

            # 读取prompt文件
            with open(shard_prompt_path_r, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                prompts_r = list(reader)
            if prompts_r:
                prompts_r = np.array(prompts_r)
                all_neutral_prompts.extend(prompts_r[:, 0].tolist())
                all_emotional_prompts.extend(prompts_r[:, 1].tolist())

        # 合并所有数据
        if all_data["neutral_prompt_feature"]:  # 检查是否有数据
            merged_data = {k: torch.cat(v) for k, v in all_data.items()}
            save_file(merged_data, DATA_CACHE_PATH)

            with open(DATA_CACHE_PATH.replace(".safetensors", ".prompt.txt"), "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t")
                for n, e in zip(all_neutral_prompts, all_emotional_prompts):
                    writer.writerow([n, e])
            logger.info(f"Rank 0: Saved merged file with {len(all_neutral_prompts)} samples to {DATA_CACHE_PATH}")

            # 删除临时文件
            for r in range(world_size):
                base_path = DATA_CACHE_PATH.replace(".safetensors", "")
                shard_path_r = f"{base_path}.rank{r}.safetensors"
                shard_prompt_path_r = f"{base_path}.rank{r}.prompt.txt"
                if os.path.exists(shard_path_r):
                    os.remove(shard_path_r)
                if os.path.exists(shard_prompt_path_r):
                    os.remove(shard_prompt_path_r)
        else:
            logger.error("No data to merge!")

    # 确保rank0完成后再退出
    if world_size > 1:
        torch.distributed.barrier()
