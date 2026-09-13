#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
JUPYTER="${PROJECT_ROOT}/.venv/bin/jupyter"
RUN_ID="${FINAL_TRAINING_RUN_ID:-manual}"
RUN_ROOT="${PROJECT_ROOT}/final_training_results/${RUN_ID}"
EXECUTED_DIR="${RUN_ROOT}/executed_notebooks"
LOG_DIR="${RUN_ROOT}/notebook_logs"
STATUS_FILE="${RUN_ROOT}/notebook_status.tsv"

NOTEBOOKS=(
  "00_data_analysis.ipynb"
  "01_baseline_object_mask.ipynb"
  "02_fixed_uvd.ipynb"
  "03_query_gated_uvd.ipynb"
  "04_rotation_consistency.ipynb"
  "05_geometry_branch_dropout.ipynb"
  "06_final_comparison_and_model_selection.ipynb"
)

mkdir -p -- "${EXECUTED_DIR}" "${LOG_DIR}"
if [[ ! -x "${JUPYTER}" ]]; then
  echo "Missing Jupyter executable: ${JUPYTER}" >&2
  exit 1
fi
if [[ ! -f "${STATUS_FILE}" ]]; then
  printf 'notebook\tstatus\tstarted_at\tfinished_at\n' > "${STATUS_FILE}"
fi

echo "Project: ${PROJECT_ROOT}"
echo "Training run ID: ${RUN_ID}"
echo "Results: ${RUN_ROOT}"
echo "Points: ${PROJECT_ROOT}/trained_points/${RUN_ID}"

for notebook_name in "${NOTEBOOKS[@]}"; do
  source_path="${PROJECT_ROOT}/final_training_notebooks/${notebook_name}"
  log_path="${LOG_DIR}/${notebook_name%.ipynb}.log"
  done_marker="${RUN_ROOT}/${notebook_name%.ipynb}.done"
  if [[ -f "${done_marker}" ]]; then
    echo "Skipping completed notebook: ${notebook_name}"
    continue
  fi
  started_at="$(date --iso-8601=seconds)"
  echo "[$(date '+%F %T')] Starting ${notebook_name}"
  set +e
  "${JUPYTER}" nbconvert \
    --to notebook \
    --execute "${source_path}" \
    --output "${notebook_name}" \
    --output-dir "${EXECUTED_DIR}" \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=-1 \
    > "${log_path}" 2>&1
  exit_code=$?
  set -e
  finished_at="$(date --iso-8601=seconds)"
  if [[ ${exit_code} -ne 0 ]]; then
    printf '%s\tfailed\t%s\t%s\n' "${notebook_name}" "${started_at}" "${finished_at}" >> "${STATUS_FILE}"
    echo "FAILED: ${notebook_name}. See ${log_path}" >&2
    exit "${exit_code}"
  fi
  touch -- "${done_marker}"
  printf '%s\tcompleted\t%s\t%s\n' "${notebook_name}" "${started_at}" "${finished_at}" >> "${STATUS_FILE}"
  echo "[$(date '+%F %T')] Completed ${notebook_name}"
done

echo "All full-training notebooks completed successfully."
echo "Best deployment checkpoint: ${PROJECT_ROOT}/trained_points/${RUN_ID}/best_model.pt"
