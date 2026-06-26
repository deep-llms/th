#!/bin/bash
#
# Download miniconda + create embeddings_hub conda env with all dependencies.
# Fully non-interactive (auto-accepts all prompts).
#
# Usage:
#   bash scripts/setup_env.sh
#

set -e

MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
INSTALL_DIR="$HOME/miniconda3"

# 1. Download and install miniconda
if [ ! -d "$INSTALL_DIR" ]; then
    echo "=== Downloading miniconda ==="
    wget -q "$MINICONDA_URL" -O /tmp/miniconda.sh
    echo "=== Installing miniconda to $INSTALL_DIR ==="
    bash /tmp/miniconda.sh -b -p "$INSTALL_DIR"
    rm /tmp/miniconda.sh
    echo "=== Miniconda installed ==="
else
    echo "=== Miniconda already installed at $INSTALL_DIR ==="
fi

# 2. Init conda for current shell
eval "$($INSTALL_DIR/bin/conda shell.bash hook)"
conda init bash --quiet 2>/dev/null || true

# 3. Accept conda terms (suppress future prompts)
conda config --set auto_activate_base false

# 4. Create embeddings_hub env
if conda env list | grep -q "embeddings_hub"; then
    echo "=== embeddings_hub env already exists ==="
else
    echo "=== Creating embeddings_hub env ==="
    conda create -n embeddings_hub python=3.11 -y
fi

# 5. Install dependencies
echo "=== Installing dependencies in embeddings_hub ==="
conda run -n embeddings_hub pip install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

conda run -n embeddings_hub pip install \
    transformers==5.9.0 datasets==4.8.5 accelerate==1.13.0 \
    scipy wandb pyarrow huggingface_hub openai

# 6. Create fasttext_env
if conda env list | grep -q "fasttext_env"; then
    echo "=== fasttext_env already exists ==="
else
    echo "=== Creating fasttext_env ==="
    conda create -n fasttext_env python=3.11 -y
fi

# 7. Install fasttext from source + numpy<2
echo "=== Installing fasttext from source in fasttext_env ==="
FASTTEXT_DIR="/tmp/fasttext_build"
if [ ! -d "$FASTTEXT_DIR" ]; then
    git clone https://github.com/facebookresearch/fastText.git "$FASTTEXT_DIR"
fi
conda run -n fasttext_env pip install "$FASTTEXT_DIR"
conda run -n fasttext_env pip install "numpy<2"
rm -rf "$FASTTEXT_DIR"

echo ""
echo "=== Done ==="
echo "Run: conda activate embeddings_hub"
echo "Verify: python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda)\""
