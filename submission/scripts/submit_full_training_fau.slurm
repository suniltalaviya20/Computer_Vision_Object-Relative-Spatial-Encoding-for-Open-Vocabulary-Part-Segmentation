#!/bin/bash -l
#
#SBATCH --job-name=cv-full-training
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100med:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --export=NONE
#SBATCH --output=outputs/slurm-%x-%j.out

unset SLURM_EXPORT_ENV
set -Eeuo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd -- "${PROJECT_ROOT}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1
export FINAL_TRAINING_RUN_ID="${1:-training_${SLURM_JOB_ID}}"

trap 'echo "Slurm time limit is approaching. Epoch checkpoints already saved can be resumed by resubmitting with run ID: ${FINAL_TRAINING_RUN_ID}"' USR1

if [[ ! -x .venv/bin/jupyter ]]; then
  echo "Missing .venv/bin/jupyter" >&2
  exit 1
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Run ID: ${FINAL_TRAINING_RUN_ID}"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print("PyTorch CUDA:", torch.version.cuda); print("GPU:", torch.cuda.get_device_name(0))'

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  bash scripts/run_full_training_overnight.sh

echo "Finished: $(date --iso-8601=seconds)"
