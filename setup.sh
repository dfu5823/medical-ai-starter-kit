#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# setup.sh — one-command setup for the DermMNIST workflow demo
#
# What it does:
# 1) checks that conda exists
# 2) creates/updates the conda env from environment.yml
# 3) activates env (prints instructions; activation in scripts is shell-dependent)
# 4) optional: installs wandb + logs in
#
# Usage:
#   bash setup.sh
#
# Notes:
# - On Windows: use Git Bash, WSL, or run the commands manually.
# - Conda activation inside a script can be finicky across shells; we print
#   the activation command at the end regardless.
# ------------------------------------------------------------

ENV_NAME="dermamnist-demo"
ENV_FILE="environment.yml"

echo "== DermMNIST demo setup =="
echo "Env name: ${ENV_NAME}"
echo "Env file: ${ENV_FILE}"
echo

# Check conda
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found."
  echo "Install Miniconda first, then re-run this script."
  echo "Miniconda: https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

# Ensure conda shell functions are available (important on some systems)
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create or update env
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Conda env '${ENV_NAME}' already exists. Updating (and pruning) packages..."
  conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
  echo "[INFO] Creating conda env '${ENV_NAME}'..."
  conda env create -f "${ENV_FILE}"
fi

echo
echo "[INFO] Activating environment..."
conda activate "${ENV_NAME}"

echo
echo "[INFO] Python version:"
python --version

echo
echo "[INFO] Installed key packages:"
python -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__)"
python -c "import medmnist; print('medmnist:', medmnist.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"

# Optional: W&B login prompt
echo
read -r -p "Do you want to login to Weights & Biases now? (y/N): " DO_WANDB
DO_WANDB="${DO_WANDB:-N}"

if [[ "${DO_WANDB}" == "y" || "${DO_WANDB}" == "Y" ]]; then
  echo "[INFO] Launching 'wandb login'..."
  wandb login
else
  echo "[INFO] Skipping W&B login. You can run 'wandb login' later."
fi

echo
echo "== Setup complete =="
echo "Next steps:"
echo "  conda activate ${ENV_NAME}"
echo "  python train.py --config configs/baseline.yaml"
