#!/bin/bash
#SBATCH --job-name=cv-part-zoom
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/zoom_%j.out
#SBATCH --error=logs/zoom_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

export HF_HUB_DISABLE_XET=1

mkdir -p logs

echo "Starting object-centric training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

python experiments/runners/object_centric_zoom/train_object_zoom.py

echo "Finished: $(date)"