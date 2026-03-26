#!/usr/bin/env bash
# ============================================================
# LTX-2 RunPod Setup Script
# Tested: Python >= 3.12, CUDA > 12.7, PyTorch ~= 2.7
# Recommended GPU: A100 80GB (or H100 / A6000 48GB)
# ============================================================
set -euo pipefail

# ── 0. Paths ─────────────────────────────────────────────────
REPO_DIR="/workspace/LTX-2"
MODELS_DIR="/workspace/models/ltx2"
GEMMA_DIR="/workspace/models/gemma-3-12b-it-qat-q4_0-unquantized"

# Optional: set your HuggingFace token if models are gated
# export HF_TOKEN="hf_..."

echo "=============================="
echo " LTX-2 RunPod Setup"
echo "=============================="

# ── 1. System packages ────────────────────────────────────────
apt-get update -qq && apt-get install -y -qq \
    git curl wget ffmpeg \
    python3.12 python3.12-venv python3.12-dev \
    build-essential libssl-dev

# Install uv (fast Python package manager required by LTX-2)
if ! command -v uv &>/dev/null; then
    echo "[*] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# ── 2. Clone repo ─────────────────────────────────────────────
if [ ! -d "$REPO_DIR" ]; then
    echo "[*] Cloning LTX-2 repository..."
    git clone https://github.com/Lightricks/LTX-2.git "$REPO_DIR"
else
    echo "[*] LTX-2 repo already exists, pulling latest..."
    git -C "$REPO_DIR" pull
fi

cd "$REPO_DIR"

# ── 3. Python environment ─────────────────────────────────────
echo "[*] Setting up Python environment with uv..."
uv sync --frozen

# Optional: install xformers for attention speedup
# uv sync --frozen --extra xformers

echo "[*] Virtual environment ready at $REPO_DIR/.venv"

# ── 4. Create model directories ───────────────────────────────
mkdir -p "$MODELS_DIR" "$GEMMA_DIR"

# ── 5. Download LTX-2.3 model checkpoints ────────────────────
# Source repo: https://huggingface.co/Lightricks/LTX-2.3
# Choose ONE of: dev (quality) or distilled (speed)
HF_BASE="https://huggingface.co/Lightricks/LTX-2.3/resolve/main"

echo "[*] Downloading LTX-2.3 distilled checkpoint (recommended for speed)..."
wget -q --show-progress -c \
    "${HF_BASE}/ltx-2.3-22b-distilled.safetensors" \
    -O "$MODELS_DIR/ltx-2.3-22b-distilled.safetensors"

# Uncomment for the dev (non-distilled, higher quality) checkpoint:
# echo "[*] Downloading LTX-2.3 dev checkpoint..."
# wget -q --show-progress -c \
#     "${HF_BASE}/ltx-2.3-22b-dev.safetensors" \
#     -O "$MODELS_DIR/ltx-2.3-22b-dev.safetensors"

# ── 6. Download spatial upsampler ─────────────────────────────
# Required for all two-stage pipelines
echo "[*] Downloading spatial upsampler (x2)..."
wget -q --show-progress -c \
    "${HF_BASE}/ltx-2.3-spatial-upscaler-x2-1.0.safetensors" \
    -O "$MODELS_DIR/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"

# Optional x1.5 upscaler (lighter):
# wget -q --show-progress -c \
#     "${HF_BASE}/ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors" \
#     -O "$MODELS_DIR/ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors"

# ── 7. Download distilled LoRA ────────────────────────────────
# Required for TwoStage* pipelines (NOT needed for DistilledPipeline or ICLoraPipeline)
echo "[*] Downloading distilled LoRA..."
wget -q --show-progress -c \
    "${HF_BASE}/ltx-2.3-22b-distilled-lora-384.safetensors" \
    -O "$MODELS_DIR/ltx-2.3-22b-distilled-lora-384.safetensors"

# ── 8. Download Gemma 3 text encoder ─────────────────────────
# Full repo clone needed (config + tokenizer + model weights)
echo "[*] Downloading Gemma 3 text encoder..."
pip install -q huggingface_hub[cli] 2>/dev/null || true
huggingface-cli download \
    google/gemma-3-12b-it-qat-q4_0-unquantized \
    --local-dir "$GEMMA_DIR" \
    --local-dir-use-symlinks False \
    ${HF_TOKEN:+--token "$HF_TOKEN"}

# ── 9. Summary ────────────────────────────────────────────────
echo ""
echo "=============================="
echo " Setup Complete!"
echo "=============================="
echo ""
echo "Activate env:   source $REPO_DIR/.venv/bin/activate"
echo "Working dir:    $REPO_DIR"
echo ""
echo "Model paths:"
echo "  Checkpoint:   $MODELS_DIR/ltx-2.3-22b-distilled.safetensors"
echo "  Upsampler:    $MODELS_DIR/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
echo "  LoRA:         $MODELS_DIR/ltx-2.3-22b-distilled-lora-384.safetensors"
echo "  Gemma:        $GEMMA_DIR"
echo ""
echo "See sample_commands.sh for inference examples."
