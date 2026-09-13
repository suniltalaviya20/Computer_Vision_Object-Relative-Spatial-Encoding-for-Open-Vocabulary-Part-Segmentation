# Full-training submission notebooks

This folder contains the seven reproducible final-training notebooks. The five
experiment notebooks optimize models from scratch and therefore require an
allocated CUDA GPU.

## Experimental order

1. `00_data_analysis.ipynb`
2. `01_baseline_object_mask.ipynb`
3. `02_fixed_uvd.ipynb`
4. `03_query_gated_uvd.ipynb`
5. `04_rotation_consistency.ipynb`
6. `05_geometry_branch_dropout.ipynb`
7. `06_final_comparison_and_model_selection.ipynb`

The five models use matched data, frozen DINOv2/OpenCLIP backbones, seed 42,
image size 224, and validation-only checkpoint selection. Each model uses a
physical batch of 8 with two-step accumulation (effective batch
16), trains for at most 30 epochs, at least 8 epochs, and stops after five
epochs without a validation-IoU improvement larger than 0.001.

## Submit on FAU Alex

From the repository root:

```bash
mkdir -p outputs
sbatch scripts/submit_full_training_fau.slurm
```

The default run ID is `training_JOB_ID`. Monitor it with:

```bash
squeue --me
tail -f outputs/slurm-cv-full-training-JOB_ID.out
```

Checkpoints are isolated under `trained_points/training_JOB_ID/`; numerical
results, figures, executed notebooks, and logs are isolated under
`final_training_results/training_JOB_ID/`.

FAU limits an Alex job to 24 hours. If the job reaches that limit, resubmit with
the original run ID printed in its log:

```bash
sbatch scripts/submit_full_training_fau.slurm training_ORIGINAL_JOB_ID
```

Completed notebooks are skipped and the interrupted experiment resumes from its
last completed epoch. Do not use a new run ID when resuming.

## Final model

After all five experiments finish, the comparison notebook copies the model
with the highest `validation_seen` IoU to:

```text
trained_points/training_JOB_ID/best_model.pt
```

`model_registry.json` records all experiment checkpoints so the UI can load and
compare them without retraining.

The UI can load the selected checkpoint and predict at the original image size:

```python
from final_training.inference import load_predictor

predictor = load_predictor("trained_points/training_JOB_ID/best_model.pt")
probability = predictor.predict(rgb_uint8_chw, parent_mask_hw, "wheel")
prediction = probability >= 0.5
```

The Slurm script requests one 20 GB `a100med` MIG instance and four CPU cores.
This matches FAU's `a100mig` resource rules and generally queues sooner than a
24-hour full-A100 allocation. Gradient accumulation retains the controlled
effective batch size without exceeding the smaller GPU memory budget.
