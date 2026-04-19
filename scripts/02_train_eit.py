"""
python scripts/02_train_eit.py --config=configs/eit.py:sdxl_eit_scale1_5_anchor

"""
import datetime
import gc
import logging
import os
from types import SimpleNamespace

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from safetensors import safe_open
from safetensors.torch import load_file
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config

from eit_with_anchor_and_grpo.models.eit import EmotionInjectionTransformer
from eit_with_anchor_and_grpo.utils.logging import setup_logging
from eit_with_anchor_and_grpo.utils.logging.helpers import (MetricLogger,
                                                            SmoothedValue)

logger = logging.getLogger(__name__)


def update_metrics(metric_logger: MetricLogger, logs: dict, state: SimpleNamespace):
    """更新 MetricLogger 和 Trackers"""
    metric_logger.update(**logs)
    # 记录学习率
    metric_logger.update(lr=state.optimizer.param_groups[0]["lr"])


def init_accelerator(config) -> Accelerator:
    # unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M")
    run_name = f"{config.train.run_name}_{unique_id}" if config.train.run_name else unique_id

    proj_config = ProjectConfiguration(project_dir=os.path.join(config.train.logdir, run_name), automatic_checkpoint_naming=False, total_limit=config.train.get("num_checkpoint_limit", 5))
    os.makedirs(proj_config.project_dir, exist_ok=True)

    kwargs_handlers = []
    if config.train.find_unused_parameters:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        kwargs_handlers.append(ddp_kwargs)

    accelerator = Accelerator(mixed_precision=config.train.get("mixed_precision", "no"), project_config=proj_config, kwargs_handlers=kwargs_handlers)

    # 仅主进程初始化 Trackers (如 Tensorboard/WandB)
    if accelerator.is_main_process:
        accelerator.init_trackers(project_name="ProjectName", config=config.to_dict(), init_kwargs={})

    # 初始化日志系统
    setup_logging(level=logging.INFO, output=accelerator.project_dir)

    # 设置随机种子
    set_seed(config.train.seed, device_specific=True)

    # 硬件优化设置
    # TF32 精度控制
    # - A100/H100 等 Ampere+ 显卡：启用 TF32 加速训练
    # - V100 等旧显卡：不支持 TF32，此设置无实际影响
    if config.get("allow_tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True

    return accelerator


def get_density(alist, vlist):
    data = np.vstack([np.asarray(alist).flatten(), np.asarray(vlist).flatten()]).T
    kde = KernelDensity(kernel="gaussian", bandwidth="silverman")
    kde.fit(data)
    log_density = kde.score_samples(data)
    return np.exp(log_density)


class EmotionDataset(Dataset):
    def __init__(self, data_path, indices, device, prompt_list=None, density_weights=None):
        self.data_path = data_path
        self.indices = indices
        self.device = device
        self.prompt_list = prompt_list
        self.density_weights = density_weights

        self._handle = None

    @property
    def handle(self):
        """懒惰打开文件句柄，确保在每个子进程中独立打开"""
        if self._handle is None:
            # framework="pt" 表示返回 PyTorch Tensor
            # device="cpu" 表示映射到 CPU 内存（由 OS 管理缺页中断）
            self._handle = safe_open(self.data_path, framework="pt", device="cpu")
        return self._handle

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        f = self.handle

        item = {
            "neutral_prompt_feature": f.get_slice("neutral_prompt_feature")[actual_idx],
            "neutral_pooled_feature": f.get_slice("neutral_pooled_feature")[actual_idx],
            "emotional_prompt_feature": f.get_slice("emotional_prompt_feature")[actual_idx],
            "emotional_pooled_feature": f.get_slice("emotional_pooled_feature")[actual_idx],
            "arousal": f.get_slice("arousal")[actual_idx],
            "valence": f.get_slice("valence")[actual_idx],
        }
        neutral_prompt = ""
        if self.prompt_list and actual_idx < len(self.prompt_list):
            line = self.prompt_list[actual_idx].strip()
            if "\t" in line:
                neutral_prompt = line.split("\t")[0]
            else:
                neutral_prompt = line
        density = self.density_weights[actual_idx] if self.density_weights is not None else 1.0
        out = {
            "neutral_prompt_feature": item["neutral_prompt_feature"],
            "arousal": torch.tensor([item["arousal"].item()], dtype=torch.float32),
            "valence": torch.tensor([item["valence"].item()], dtype=torch.float32),
            "emotional_prompt_feature": item["emotional_prompt_feature"],
            "neutral_prompt": neutral_prompt,
            "density": torch.tensor([density], dtype=torch.float32),
        }
        return out


def load_data_and_create_datasets(data_cache_path, prompt_cache_path, device, test_size=0.01, random_state=42):

    temp_data = load_file(data_cache_path, device="cpu")

    num_samples = len(temp_data["neutral_prompt_feature"])
    logger.info(f"Loaded metadata for {num_samples} samples")
    alist = temp_data["arousal"].numpy().flatten()
    vlist = temp_data["valence"].numpy().flatten()
    logger.info("Get density ......")
    den_list = get_density(alist, vlist)
    density_weights = 1.0 / torch.tensor(den_list, dtype=torch.float32).clamp(min=1e-6)

    # ... prompt 读取 ...
    prompt_list = None
    if os.path.exists(prompt_cache_path):
        with open(prompt_cache_path, "r") as f:
            prompt_list = f.readlines()

    # ... 划分索引 ...
    all_indices = list(range(num_samples))
    train_indices, val_indices = train_test_split(all_indices, test_size=test_size, random_state=random_state)

    del temp_data

    gc.collect()

    logger.info(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")

    # 传入文件路径而非 data_dict
    train_dataset = EmotionDataset(data_cache_path, train_indices, device, prompt_list, density_weights)
    val_dataset = EmotionDataset(data_cache_path, val_indices, device, prompt_list, density_weights)

    return train_dataset, val_dataset, density_weights


def init_state(config, accelerator):
    train_dataset, val_dataset, density_weights = load_data_and_create_datasets(config.data_cache_path, config.prompt_cache_path, accelerator.device, config.test_size, config.train.seed )
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    model_config = GPT2Config.from_pretrained(config.config_dir)
    model = EmotionInjectionTransformer(model_config, final_out_type="Linear+LN", sd_feature_dim=config.model.sd_feature_dim)
    if config.eit_ckpt and os.path.exists(config.eit_ckpt):
        state_dict = torch.load(config.eit_ckpt, map_location="cpu")
        model.load_state_dict(state_dict)
        logger.info(f"Loaded EIT checkpoint from {config.eit_ckpt}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    lr_scheduler = None
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    total_iters = config.train.epochs * len(train_loader)
    logger.info(f"total_iters: {total_iters}")
    return SimpleNamespace(
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, train_loader=train_loader, val_loader=val_loader, density_weights=density_weights, global_step=0, train_config=config.train, alpha=config.scale_factor, anchor_weight=config.anchor_weight, enable_density=config.enable_density
    )


def train_step(batch, state, accelerator):
    neutral = batch["neutral_prompt_feature"].to(accelerator.device, non_blocking=True).to(torch.float32)
    arousal = batch["arousal"].to(accelerator.device, non_blocking=True)
    valence = batch["valence"].to(accelerator.device, non_blocking=True)
    emotional = batch["emotional_prompt_feature"].to(accelerator.device, non_blocking=True).to(torch.float32)
    with accelerator.autocast():
        # 监督：EIT(neutral, a, v) -> target
        pred_emotional = state.model(inputs_embeds=neutral, arousal=arousal, valence=valence)[0]
    target = (emotional - neutral) * state.alpha + neutral
    if state.enable_density:
        density = batch["density"].squeeze(1).to(accelerator.device)
        density = density.view(-1, 1, 1)
        target = target / density
        pred_emotional = pred_emotional / density

    loss_sup = torch.nn.functional.mse_loss(pred_emotional, target, reduction="none").mean(dim=[1, 2])

    if state.anchor_weight > 0:
        with accelerator.autocast() and torch.no_grad():
            pred_anchor = state.model(inputs_embeds=neutral, arousal=torch.zeros_like(arousal), valence=torch.zeros_like(valence))[0]
        loss_anchor = torch.nn.functional.mse_loss(pred_anchor, neutral, reduction="none").mean(dim=[1, 2])
        loss = loss_sup + state.anchor_weight * loss_anchor
    else:
        loss = loss_sup

    loss = loss.mean()
    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(state.model.parameters(), state.train_config.max_grad_norm)
    state.optimizer.step()
    state.optimizer.zero_grad()

    return {"loss": loss.item(), "loss_sup": loss_sup.mean().item(), "loss_anchor": loss_anchor.mean().item() if state.anchor_weight > 0 else 0.0}


def evaluate(state, accelerator):
    state.model.eval()
    total_sum = 0.0  # 全局 loss 总和
    total_num = 0  # 全局样本数
    with torch.no_grad():
        for batch in state.val_loader:
            neutral = batch["neutral_prompt_feature"].to(accelerator.device, non_blocking=True).to(torch.float32)
            arousal = batch["arousal"].to(accelerator.device, non_blocking=True)
            valence = batch["valence"].to(accelerator.device, non_blocking=True)
            emotional = batch["emotional_prompt_feature"].to(accelerator.device, non_blocking=True).to(torch.float32)
            with accelerator.autocast():
                pred_emotional = state.model(inputs_embeds=neutral, arousal=arousal, valence=valence)[0]

            target = (emotional - neutral) * state.alpha + neutral
            if state.enable_density:
                density = batch["density"].squeeze(1).to(accelerator.device)
                density = density.view(-1, 1, 1)
                target = target / density
                pred_emotional = pred_emotional / density
            loss_sup = torch.nn.functional.mse_loss(pred_emotional, target, reduction="none").mean(dim=[1, 2])
            if state.anchor_weight > 0:
                with accelerator.autocast():
                    pred_anchor = state.model(inputs_embeds=neutral, arousal=torch.zeros_like(arousal), valence=torch.zeros_like(valence))[0]
                loss_anchor = torch.nn.functional.mse_loss(pred_anchor, neutral, reduction="none").mean(dim=[1, 2])
                loss = loss_sup + state.anchor_weight * loss_anchor
            else:
                loss = loss_sup
            # 1) 每张卡内部先不求平均，只求和
            loss_sum = loss.sum()  # 该卡 batch 的 loss 总和
            bs = batch["arousal"].size(0)  # 该卡 batch 的样本数

            # 2) 跨卡汇总
            loss_sum, bs = accelerator.gather((loss_sum, torch.tensor(bs, device=loss_sum.device)))

            if accelerator.is_main_process:
                total_sum += loss_sum.sum().item()
                total_num += bs.sum().item()

    state.model.train()
    # 3) 主进程返回全局平均，其它卡返回 None
    return total_sum / total_num if accelerator.is_main_process else None


def main(config):
    accelerator = init_accelerator(config)
    state = init_state(config, accelerator)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", SmoothedValue(window_size=20, fmt="{global_avg:.4f}"))
    metric_logger.add_meter("lr", SmoothedValue(window_size=20, fmt="{value:.6f}"))
    metric_logger.add_meter("loss_sup", SmoothedValue(window_size=20, fmt="{global_avg:.4f}"))
    metric_logger.add_meter("loss_anchor", SmoothedValue(window_size=20, fmt="{global_avg:.4f}"))
    state.model.train()

    best_val_loss = float("inf")

    for epoch in range(config.train.epochs):
        for batch in metric_logger.log_every(state.train_loader, config.train.print_freq, header=f"Train [{epoch}/{config.train.epochs}] "):
            if state.lr_scheduler and state.global_step > 0:
                state.lr_scheduler.step()
            logs = train_step(batch, state, accelerator)
            update_metrics(metric_logger, logs, state)
            state.global_step += 1

        val_loss = evaluate(state, accelerator)

        # 2) 主进程打印 & 判断 best
        if accelerator.is_main_process:
            logger.info(f"[epoch {epoch}] Val loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 3) 保存当前最优权重
                accelerator.save_model(model=state.model, save_directory=os.path.join(accelerator.project_dir, "checkpoint_best"))
                logger.info("New best val_loss -> saved checkpoint_best")

        # ---------- 定期/最后一个 epoch 的 checkpoint ----------
        if epoch and ((epoch + 1) % config.train.save_freq == 0 or (epoch + 1) == config.train.epochs):
            if accelerator.is_main_process:
                accelerator.save_model(model=state.model, save_directory=os.path.join(accelerator.project_dir, f"checkpoint_epoch-{epoch}"))

    accelerator.end_training()


if __name__ == "__main__":
    from absl import app, flags
    from ml_collections import config_flags

    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file("config", "configs/eit.py", "Configuration file.")

    def run(_):
        main(FLAGS.config)

    app.run(run)
