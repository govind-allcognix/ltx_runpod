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
# uv installs to ~/.local/bin on Linux; source its env file so the current
# shell session can find it immediately without needing a new login shell.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv &>/dev/null; then
    echo "[*] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the env file the installer generates (sets PATH correctly)
    if [ -f "$HOME/.local/bin/env" ]; then
        # shellcheck source=/dev/null
        source "$HOME/.local/bin/env"
    elif [ -f "$HOME/.cargo/env" ]; then
        # shellcheck source=/dev/null
        source "$HOME/.cargo/env"
    fi
    # Hard-set PATH as a final fallback
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# Verify uv is now available
if ! command -v uv &>/dev/null; then
    echo "[ERROR] uv still not found after installation. PATH=$PATH"
    echo "Try running: export PATH=\$HOME/.local/bin:\$PATH && bash setup.sh"
    exit 1
fi
echo "[*] uv found: $(uv --version)"

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

# ── 5. Install huggingface_hub + hf_transfer for fast downloads ──
# hf_transfer is a Rust-based downloader from HF — much faster than wget.
# HF_HUB_ENABLE_HF_TRANSFER=1 activates it automatically for all hf commands.
echo "[*] Installing huggingface_hub with hf_transfer..."
python3 -m pip install -q "huggingface_hub[hf_transfer,cli]" hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Use 'python3 -m huggingface_hub' instead of 'huggingface-cli'.
# This avoids PATH issues: the module is guaranteed to be importable
# by the same Python that just installed it, regardless of shell PATH.
hf_download_file() {
    local repo="$1" filename="$2" dest="$3"
    if [ -f "$dest" ]; then
        echo "[skip] $filename already exists."
        return
    fi
    echo "[*] Downloading $filename ..."
    python3 -m huggingface_hub download "$repo" "$filename" \
        --local-dir "$(dirname "$dest")" \
        --local-dir-use-symlinks False \
        ${HF_TOKEN:+--token "$HF_TOKEN"}
}

# ── 6. Download LTX-2.3 model checkpoint ─────────────────────
# Source repo: https://huggingface.co/Lightricks/LTX-2.3
# Choose ONE of: distilled (speed/default) or dev (quality)
hf_download_file \
    Lightricks/LTX-2.3 \
    ltx-2.3-22b-distilled.safetensors \
    "$MODELS_DIR/ltx-2.3-22b-distilled.safetensors"

# Uncomment for the dev (non-distilled, higher quality) checkpoint:
# hf_download_file \
#     Lightricks/LTX-2.3 \
#     ltx-2.3-22b-dev.safetensors \
#     "$MODELS_DIR/ltx-2.3-22b-dev.safetensors"

# ── 7. Download spatial upsampler ─────────────────────────────
# Required for all two-stage pipelines
hf_download_file \
    Lightricks/LTX-2.3 \
    ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
    "$MODELS_DIR/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"

# Optional x1.5 upscaler (lighter VRAM):
# hf_download_file \
#     Lightricks/LTX-2.3 \
#     ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors \
#     "$MODELS_DIR/ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors"

# ── 8. Download distilled LoRA ────────────────────────────────
# Required for TwoStage* pipelines (NOT needed for DistilledPipeline or ICLoraPipeline)
hf_download_file \
    Lightricks/LTX-2.3 \
    ltx-2.3-22b-distilled-lora-384.safetensors \
    "$MODELS_DIR/ltx-2.3-22b-distilled-lora-384.safetensors"

# ── 9. Download Gemma 3 text encoder ─────────────────────────
# Pulls all files in the repo (config + tokenizer + weights)
echo "[*] Downloading Gemma 3 text encoder..."
python3 -m huggingface_hub download \
    google/gemma-3-12b-it-qat-q4_0-unquantized \
    --local-dir "$GEMMA_DIR" \
    --local-dir-use-symlinks False \
    ${HF_TOKEN:+--token "$HF_TOKEN"}

# ── 10. Summary ───────────────────────────────────────────────
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
