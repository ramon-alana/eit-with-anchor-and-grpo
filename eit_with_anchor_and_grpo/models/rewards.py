from functools import partial
from typing import Literal

import clip
import numpy as np
import torch
from PIL import Image
from torchmetrics.multimodal.clip_iqa import CLIPImageQualityAssessment
from torchmetrics.multimodal.clip_score import CLIPScore


class CLIPRegressor(torch.nn.Module):
    """VA预测器的回归头"""

    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Sequential(torch.nn.Linear(in_features=512, out_features=1, bias=True), torch.nn.Sigmoid())

    def forward(self, x):
        return self.classifier(x)


class VAPredictor(torch.nn.Module):
    """
    基于CLIP的VA预测器
    输出范围: [-3, 3]（论文采用的V-A空间范围）
    """

    def __init__(self, arousal_ckpt_path, valence_ckpt_path, device="cuda"):
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

        # Arousal预测器
        self.ar = CLIPRegressor()
        self.ar.load_state_dict(torch.load(arousal_ckpt_path, map_location=device))

        # Valence预测器
        self.vr = CLIPRegressor()
        self.vr.load_state_dict(torch.load(valence_ckpt_path, map_location=device))

        self.to(device)
        self.eval()

    @torch.no_grad()
    def forward(self, image_pil, device):
        """
        Args:
            image_pil: PIL.Image
        Returns:
            (valence, arousal): 范围[-3, 3]
        """
        image_input = self.preprocess(image_pil).unsqueeze(0).to(device)
        image_features = self.model.encode_image(image_input).float()

        # Sigmoid输出[0,1] → 映射到[-3,3]
        v = self.vr(image_features) * 6 - 3
        a = self.ar(image_features) * 6 - 3

        return v.item(), a.item()


class EmotiCrafterEvaluator:
    """
    论文Table 1指标计算：V-Error, A-Error, CLIPScore, CLIP-IQA
    """

    def __init__(self, arousal_ckpt, valence_ckpt, device="cuda", model_name_or_path="/WEIGHTS/WZX/openai/clip-vit-large-patch14", model_name_or_path_2=None):
        self.device = device

        # VA预测器（用于V/A-Error）
        self.va_predictor = VAPredictor(arousal_ckpt, valence_ckpt, device)

        # CLIPScore（图文相似度）- 第一个模型
        self.clip_score_metric = CLIPScore(model_name_or_path=model_name_or_path)
        self.clip_score_metric.to(device)

        # CLIP-IQA（图像质量评估，无参考）- 第一个模型
        self.clip_iqa_metric = CLIPImageQualityAssessment(model_name_or_path=model_name_or_path)
        self.clip_iqa_metric.to(device)

        # 第二个CLIP模型（如果提供）
        self.has_second_model = model_name_or_path_2 is not None
        if self.has_second_model:
            self.clip_score_metric2 = CLIPScore(model_name_or_path=model_name_or_path_2)
            self.clip_score_metric2.to(device)

            self.clip_iqa_metric2 = CLIPImageQualityAssessment(model_name_or_path=model_name_or_path_2)
            self.clip_iqa_metric2.to(device)

    @torch.no_grad()
    def calculate_clip_score(self, generated_image_pil, neutral_prompt, use_second_model=False):
        """
        计算CLIPScore（图像与中性提示词的相似度）

        Args:
            generated_image_pil: PIL.Image，生成的图像
            neutral_prompt: str, 中性提示词
            use_second_model: bool, 是否使用第二个模型

        Returns:
            float: CLIPScore（范围约0-100）
        """
        image_np = np.array(generated_image_pil).astype(np.uint8)

        if len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=-1)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

        if use_second_model and self.has_second_model:
            score = self.clip_score_metric2(image_tensor, [neutral_prompt])
        else:
            score = self.clip_score_metric(image_tensor, [neutral_prompt])

        return score.item()

    @torch.no_grad()
    def calculate_clip_iqa(self, generated_image_pil, use_second_model=False):
        """
        计算CLIP-IQA（无参考图像质量评估）

        Args:
            generated_image_pil: PIL.Image，生成的图像
            use_second_model: bool, 是否使用第二个模型

        Returns:
            float: CLIP-IQA分数（越高越好，通常范围有变化）
        """
        image_np = np.array(generated_image_pil).astype(np.uint8)

        if len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=-1)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)

        if use_second_model and self.has_second_model:
            scores_dict = self.clip_iqa_metric2(image_tensor)
        else:
            scores_dict = self.clip_iqa_metric(image_tensor)

        if isinstance(scores_dict, dict):
            score = torch.mean(torch.stack(list(scores_dict.values()))).item()
        else:
            score = scores_dict.item()

        return score

    @torch.no_grad()
    def calculate_va_error(self, generated_image_pil, target_v, target_a):
        """
        计算V-Error和A-Error

        Args:
            generated_image_pil: PIL.Image，生成的图像
            target_v: float, 目标Valence值 [-3, 3]
            target_a: float, 目标Arousal值 [-3, 3]

        Returns:
            dict: {'v_error': float, 'a_error': float}
        """
        pred_v, pred_a = self.va_predictor(generated_image_pil, self.device)

        v_error = abs(pred_v - target_v)
        a_error = abs(pred_a - target_a)

        return {"v_error": v_error, "a_error": a_error, "pred_v": pred_v, "pred_a": pred_a}

    def evaluate(self, generated_image_pil, target_v, target_a, neutral_prompt):
        """
        一次性计算论文Table 1所有指标

        Args:
            generated_image_pil: PIL.Image
            target_v: float, 目标Valence [-3, 3]
            target_a: float, 目标Arousal [-3, 3]
            neutral_prompt: str, 输入的中性提示词

        Returns:
            dict: 包含v_error, a_error, clip_score, clip_iqa
        """
        va_result = self.calculate_va_error(generated_image_pil, target_v, target_a)
        clip_score = self.calculate_clip_score(generated_image_pil, neutral_prompt)
        clip_iqa = self.calculate_clip_iqa(generated_image_pil)

        result = {"v_error": va_result["v_error"], "a_error": va_result["a_error"], "clip_score": clip_score, "clip_iqa": clip_iqa, "pred_v": va_result["pred_v"], "pred_a": va_result["pred_a"]}

        # 如果有第二个模型，添加第二个模型的指标
        if self.has_second_model:
            clip_score2 = self.calculate_clip_score(generated_image_pil, neutral_prompt, use_second_model=True)
            clip_iqa2 = self.calculate_clip_iqa(generated_image_pil, use_second_model=True)

            result.update({"clip_score2": clip_score2, "clip_iqa2": clip_iqa2})

        return result


# ── VA reward ──────────────────────────────────────────────────────────────────


class VARewardModel:
    """
    Wraps VAPredictor to produce scalar rewards for a batch of generated images.
    reward = -(w_v·|pred_v - tgt_v| + w_a·|pred_a - tgt_a|)
    """

    def __init__(self, arousal_ckpt: str, valence_ckpt: str, device: str, w_valence: float = 1.0, w_arousal: float = 1.0):
        self.predictor = VAPredictor(arousal_ckpt, valence_ckpt, device=device)
        self.predictor.eval()
        self.device = device
        self.w_valence = w_valence
        self.w_arousal = w_arousal

    @torch.no_grad()
    def __call__(self, pil: Image, metadata: dict) -> torch.Tensor:  # [B, 3, H, W] float32 in [0,1]  # list of dicts with 'arousal', 'valence'  # [B] rewards
        rewards = []
        # pil = pil.resize((512, 512))  # 1024 原始大小
        tgt_v = metadata["valence"].item()
        tgt_a = metadata["arousal"].item()
        pred_v, pred_a = self.predictor(pil, self.device)
        rewards.append(-(self.w_valence * abs(pred_v - tgt_v) + self.w_arousal * abs(pred_a - tgt_a)))
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)


class MutiRewardModel:
    def __init__(self, arousal_ckpt: str, valence_ckpt: str, device: str, clip_model_name: str = "openai/clip-vit-base-patch32", w_valence: float = 1.0, w_arousal: float = 1.0, w_clip: float = 1.0, reward_mode: Literal["va", "clip", "all"] = "all", va_norm: float = 6, clip_norm: float = 100):
        """
        Args:
            arousal_ckpt: Path to arousal predictor checkpoint
            valence_ckpt: Path to valence predictor checkpoint
            device: Device to run models on
            clip_model_name: CLIP model name for clip score
            w_valence: Weight for valence error (negative reward)
            w_arousal: Weight for arousal error (negative reward)
            w_clip: Weight for CLIP score (positive reward)
            reward_mode: Which rewards to compute ("va", "clip", or "all")
        """
        self.device = device
        self.reward_mode = reward_mode
        self.w_valence = w_valence
        self.w_arousal = w_arousal
        self.w_clip = w_clip
        self.va_norm = va_norm
        self.clip_norm = clip_norm

        # Initialize models based on reward_mode
        if reward_mode in ("va", "all"):
            self.predictor = VAPredictor(arousal_ckpt, valence_ckpt, device=device)
        else:
            self.predictor = None

        if reward_mode in ("clip", "all"):
            self.clip_score_metric = CLIPScore(model_name_or_path=clip_model_name)
            self.clip_score_metric.to(device)
        else:
            self.clip_score_metric = None

    @torch.no_grad()
    def __call__(self, pil: Image.Image, metadata: dict) -> torch.Tensor:
        """
        Args:
            pil: PIL Image (already resized to 512x512)
            metadata: Dict with 'arousal', 'valence', 'prompt' keys
        Returns:
            Scalar reward tensor
        """
        total_reward = torch.tensor(0.0, device=self.device)

        # VA reward: negative error (lower is better, so we negate)
        if self.reward_mode in ("va", "all"):
            assert self.predictor is not None
            pred_v, pred_a = self.predictor(pil, self.device)  # Returns Python floats

            # Ensure tensors on correct device (scalars default to CPU)
            pred_v = torch.tensor(pred_v, device=self.device)
            pred_a = torch.tensor(pred_a, device=self.device)

            tgt_v = metadata["valence"]
            tgt_a = metadata["arousal"]

            # Convert targets to tensors on same device
            tgt_v = torch.as_tensor(tgt_v, device=self.device)
            tgt_a = torch.as_tensor(tgt_a, device=self.device)

            va_error = self.w_valence * torch.abs(pred_v - tgt_v) + self.w_arousal * torch.abs(pred_a - tgt_a)
            va_reward = -va_error / self.va_norm

            total_reward = total_reward + va_reward

        # CLIP reward: higher is better
        if self.reward_mode in ("clip", "all"):
            assert self.clip_score_metric is not None

            # Convert PIL to tensor [1, 3, H, W] in [0, 255] -> CLIP expects [0, 255]
            image_np = np.array(pil).astype(np.uint8)  # [H, W, 3]
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(self.device)  # [1, 3, H, W]

            # CLIPScore expects uint8 [0, 255] and list of texts
            prompt = metadata.get("prompt", "")
            # Repeat image for 2 comparisons (CLIPScore internal requirement)
            clip_score = self.clip_score_metric(image_tensor, [prompt])

            # Normalize CLIP score (typically 0-100) and apply weight
            clip_reward = self.w_clip * clip_score / self.clip_norm  # Scale to ~0-1 range
            total_reward = total_reward + clip_reward

        return total_reward


if __name__ == "__main__":
    # HF_HOME=/WEIGHTS/WZX
    reward_model = MutiRewardModel("metrics/arousal1_CLIP_lr=0.001_loss=MSELoss_sc=test_cuda-1.pth", "metrics/valence1_CLIP_lr=0.0001_loss=MSELoss_sc=test_cuda.pth", device="cuda", reward_mode="all", w_clip=1.0, w_valence=0.25, w_arousal=0.25)

    pil = Image.open("results/V-A (-2.0,2.5) | A man is running fast.png").convert("RGB")
    va_score = reward_model(pil, {"arousal": torch.Tensor([-2]), "valence": torch.Tensor([2.5]), "prompt": "A man is running fast"})
    print(va_score)
