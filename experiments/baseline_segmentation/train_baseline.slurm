#!/bin/bash -l

#SBATCH --job-name=cv-part-baseline
#SBATCH --gres=gpu:a40:1
#SBATCH --time=12:00:00
#SBATCH --export=NONE
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

unset SLURM_EXPORT_ENV


set -e


cd "$HOME/Computer_Vision_Object-Relative-Spatial-Encoding-for-Open-Vocabulary-Part-Segmentation"


module load python/3.12-base
module load cuda/12.8.1


source .venv/bin/activate

export HF_HUB_DISABLE_XET=1

mkdir -p logs


echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "========================================"


python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY


echo "========================================"
echo "Starting baseline training"
echo "========================================"


python experiments/baseline_segmentation/train_baseline.py