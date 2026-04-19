# eit-with-anchor-and-grpo
## 🛠️ Setup Guide
### Prerequisites
- Python 3.10
- CUDA 12.1

### 1. Environment Configuration
```bash
# Configure PyPI mirror (recommended for users in Chinese mainland)
export UV_INDEX_URL="https://repo.huaweicloud.com/repository/pypi/simple/"
export HF_HOME=/WEIGHTS/WZX && mkdir -p $HF_HOME

# Install uv and build dependencies
pip install uv -i ${UV_INDEX_URL}
uv pip install "setuptools<75.3.0" cython -i ${UV_INDEX_URL}

# Install PyTorch 2.3.1 (CUDA 12.1)
uv pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
uv pip install -r docker/req.txt --system --no-build-isolation -i ${UV_INDEX_URL}

# Install additional libraries
uv pip install xformers==0.0.27 --index-url https://download.pytorch.org/whl/cu121
uv pip install trl==0.22.0  ml_collections --system --no-deps
```

### 2. Environment Verification
```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## 🤗 Hugging Face Model Preparation
Download all required pretrained models to the target directory:
```bash
hf download openai/clip-vit-base-patch32 --local-dir /WEIGHTS/WZX/openai/clip-vit-base-patch32
hf download openai/clip-vit-large-patch14 --local-dir /WEIGHTS/WZX/openai/clip-vit-large-patch14
hf download hf-internal-testing/tiny-sd3-pipe --local-dir /WEIGHTS/WZX/tiny-sd3-pipe
hf download stabilityai/stable-diffusion-3.5-medium --local-dir /WEIGHTS/WZX/stabilityai/stable-diffusion-3.5-medium
```

---

## 🚀 Training Pipeline
### Step 1: Train EIT Model
```bash
# Initialize environment variables
source scripts/envs/setup_env.sh

# Preprocess dataset (Debug mode)
python scripts/01_preprocess_sd3.py --MODEL_PATH=/WEIGHTS/WZX/tiny-sd3-pipe --DATA_CACHE_PATH=data/data-sd3-fp16.safetensors --DEBUG=True

# Preprocess dataset (Full model)
python scripts/01_preprocess_sd3.py --MODEL_PATH=/WEIGHTS/WZX/stabilityai/stable-diffusion-3.5-medium --DATA_CACHE_PATH=data/data-sd3-fp16.safetensors

# Train EIT model (Debug, single GPU)
python scripts/02_train_eit.py --config=configs/eit.py:sd3tiny_scale1_1_anchor

# Train EIT model (Full, 8-GPU distributed)
torchrun --nproc_per_node=8 scripts/02_train_eit.py --config=configs/eit.py:sd35M_scale1_1_anchor
```

### Step 2: Train GRPO Model
First, modify the EIT checkpoint path in the config file:
```python
# Update in configs/grpo.py:va_pickscore_sd3tiny_scale1_1_anchor_fast_nocfg_gpux1
config.eit.eit_ckpt="/logs/wzx_data/sd3tiny_scale1_1_anchor/sd3tiny_scale1_1_anchor_2026.04.19_09.23/checkpoint_best/model.safetensors"
```

Start training:
```bash
# Train GRPO (Debug, single GPU)
torchrun --nproc_per_node=1 scripts/03_train_sd3_fast.py --config=configs/grpo.py:va_pickscore_sd3tiny_scale1_1_anchor_fast_nocfg_gpux1

# Train GRPO (Full, 8-GPU distributed)
torchrun --nproc_per_node=8 scripts/03_train_sd3_fast.py --config=configs/grpo.py:pickscore_sd3_fast_nocfg_fp16_v100x8
```

---

## 🖼️ Inference & Evaluation
### Test Inference (5x5 Grid Output)
```bash
torchrun --nproc_per_node=1 scripts/04_1_inference_sd3_5x5_from_csv.py configs/infer/sd3_tiny_grpo.py
```

---

## 📚 Citation
```
```

