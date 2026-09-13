#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
JUPYTER="${PROJECT_ROOT}/.venv/bin/jupyter"
NOTEBOOK_DIR="${PROJECT_ROOT}/final_training_notebooks"
RUN_ID="${FINAL_TRAINING_RUN_ID:-manual}"
RUN_ROOT="${PROJECT_ROOT}/training_results"
EXECUTED_DIR="${RUN_ROOT}/executed_notebooks"

NOTEBOOKS=(
  "00_data_analysis.ipynb"
  "01_baseline_object_mask.ipynb"
  "02_fixed_uvd.ipynb"
  "03_query_gated_uvd.ipynb"
  "04_rotation_consistency.ipynb"
  "05_geometry_branch_dropout.ipynb"
  "06_final_comparison_and_model_selection.ipynb"
)

mkdir -p -- "${EXECUTED_DIR}"
if [[ ! -x "${JUPYTER}" ]]; then
  echo "Missing Jupyter executable: ${JUPYTER}" >&2
  exit 1
fi
echo "Project: ${PROJECT_ROOT}"
echo "Training run ID: ${RUN_ID}"
echo "Results: ${RUN_ROOT}"
echo "Checkpoints: ${RUN_ROOT}"

for notebook_name in "${NOTEBOOKS[@]}"; do
  source_path="${NOTEBOOK_DIR}/${notebook_name}"
  echo "[$(date '+%F %T')] Starting ${notebook_name}"
  "${JUPYTER}" nbconvert \
    --to notebook \
    --execute "${source_path}" \
    --output "${notebook_name}" \
    --output-dir "${EXECUTED_DIR}" \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=-1
  echo "[$(date '+%F %T')] Completed ${notebook_name}"
done

echo "All full-training notebooks completed successfully."
echo "Best UI checkpoint: ${RUN_ROOT}/best_model.pt"
