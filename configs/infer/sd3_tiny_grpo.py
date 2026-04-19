MODEL_TYPE = "sd3"
SD3_PATH = "/WEIGHTS/WZX/tiny-sd3-pipe/"
OUTPUT_DIR = "results/sd3_tiny_grpo.py"
HEIGHT = 32  # FLUX和OmniGen2默认1024
WIDTH = 32
NUM_INFERENCE_STEPS = 28  # SD3 默认50 
GUIDANCE_SCALE = 4.5  # SD3 默认3.5
USE_EMO_PROMPT=False
EIT_CKPT="/logs/wzx_data/sd3tiny_scale1_1_anchor/sd3tiny_scale1_1_anchor_2026.04.19_09.23/checkpoint_best/model.safetensors"
LORA_PATH="/logs/wzx_data/va_pickscore_sd3tiny_scale1_1_anchor_fast_nocfg_gpux1/ckpt/checkpoints/checkpoint-200/lora"

# torchrun --nproc_per_node=1  scripts/04_1_inference_sd3_5x5_from_csv.py configs/infer/sd3_tiny_grpo.py