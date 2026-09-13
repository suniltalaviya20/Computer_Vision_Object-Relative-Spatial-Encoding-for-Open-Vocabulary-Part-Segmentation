# Final Professor Submission

## Project

**Object-Relative Spatial Encoding for Open-Vocabulary Part Segmentation**

The model receives an RGB image, a binary parent-object mask, and a text part
query. It predicts the requested object-part mask. Experiments use
Pascal-Part-116 with frozen DINOv2 ViT-S/14 and OpenCLIP ViT-B/32 QuickGELU
encoders.

## Submitted experiment

The completed run is `training_4217799`, executed on 11–12 September 2026.
All seven notebooks completed successfully. Five matched-capacity models were
trained with seed 42 and selected using `validation_seen` IoU only.

| Model | Epoch | Validation IoU | Test seen IoU | Test unseen IoU |
|---|---:|---:|---:|---:|
| Object-mask baseline | 14 | 0.2814 | 0.2935 | 0.2442 |
| Fixed UVD | 12 | 0.2855 | 0.2984 | 0.2570 |
| Query-gated UVD | 12 | 0.2837 | 0.2952 | 0.2518 |
| **Rotation-consistent UVD** | **21** | **0.2981** | **0.3096** | **0.2721** |
| Geometry-dropout UVD | 15 | 0.2871 | 0.3013 | 0.2468 |

The selected deployment model is rotation-consistent UVD. Test metrics were
reported only after checkpoint selection.

## Submission contents

```text
submission/
├── README.md
├── CHECKSUMS.sha256              # integrity hashes for submitted files
├── requirements.txt
├── datasets/                    # Pascal-Part-116 loader and metadata
├── final_training/              # architecture, training, and inference
├── final_training_notebooks/    # seven ordered source notebooks
├── final_training_results/
│   └── training_4217799/        # metrics, plots, logs, executed notebooks
├── trained_points/
│   └── training_4217799/        # checkpoints and model registry
└── scripts/                     # preparation and FAU execution scripts
```

The dataset and frozen backbone weights are intentionally excluded because of
their size. Training-resume checkpoints (`best.pt` and `last.pt`) are included
for reproducibility. Compact `ui_model.pt` files are included for inference.

## Key files for evaluation

- `final_training_results/training_4217799/all_experiment_comparison.csv`
- `final_training_results/training_4217799/final_model_comparison.png`
- `final_training_results/training_4217799/notebook_status.tsv`
- `final_training_results/training_4217799/executed_notebooks/`
- `trained_points/training_4217799/model_registry.json`
- `trained_points/training_4217799/best_model.pt`

Per-experiment folders contain configuration, training history, summary
metrics, detailed predictions, qualitative results, training curves, completion
metadata, and an artifact manifest.

From the repository root, verify that the submission files were not corrupted:

```bash
sha256sum --check submission/CHECKSUMS.sha256
```

## Environment

Use Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset preparation

From the `submission/` directory:

```bash
python scripts/download_dataset.py
python scripts/prepare_dataset.py
```

Expected dataset location:

```text
data/raw/PascalPart116/
```

Prepared manifests and split files are written to `data/processed/` and
`data/splits/`. Dataset files are not included in the submission archive.

## Reproduce the complete training run

The ordered notebooks are:

1. `00_data_analysis.ipynb`
2. `01_baseline_object_mask.ipynb`
3. `02_fixed_uvd.ipynb`
4. `03_query_gated_uvd.ipynb`
5. `04_rotation_consistency.ipynb`
6. `05_geometry_branch_dropout.ipynb`
7. `06_final_comparison_and_model_selection.ipynb`

On FAU Alex, from the `submission/` directory:

```bash
mkdir -p outputs
sbatch scripts/submit_full_training_fau.slurm
```

The job prints its run ID and writes checkpoints to `trained_points/<run-id>/`
and evidence to `final_training_results/<run-id>/`. If the job reaches the
24-hour limit, resubmit with the same run ID so completed notebooks are skipped
and interrupted training resumes:

```bash
sbatch scripts/submit_full_training_fau.slurm training_ORIGINAL_JOB_ID
```

## Inference

The compact selected checkpoint is:

```text
trained_points/training_4217799/best_model.pt
```

The inference API expects a `uint8` RGB tensor `[3, H, W]`, a binary parent mask
`[H, W]`, and a text query:

```python
from final_training.inference import load_predictor

predictor = load_predictor(
    "trained_points/training_4217799/best_model.pt"
)
probability = predictor.predict(rgb_image, parent_mask, "wheel")
prediction = probability >= 0.5
```

For offline execution, provide the exact frozen dependencies at:

```text
dinov2/
models/pretrained/dinov2_vits14_pretrain.pth
models/pretrained/open_clip/ViT-B-32.pt
```

These large third-party files are not part of the submission.
