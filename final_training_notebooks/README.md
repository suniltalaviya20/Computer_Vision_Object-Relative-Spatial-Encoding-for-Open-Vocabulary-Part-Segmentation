# Full-training submission notebooks

This folder contains the seven reproducible final-training notebooks. The five
experiment notebooks optimize models from scratch and therefore require an
allocated CUDA GPU.

Each experiment notebook contains only the material needed for evaluation:
the selected configuration, complete dataset/model/loss/training implementation,
training command, complete epoch history, result tables, figures, and
saved-output locations. The notebooks use the root dataset package and contain
the complete training implementation directly in their cells.
The final notebook compares all five models and records the validation-selected
local-UI checkpoint.

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

The first experiment downloads DINOv2 ViT-S/14 and OpenCLIP ViT-B/32
QuickGELU (`openai`) from their official online sources. Subsequent notebooks
reuse the standard local caches automatically.

## Run locally

Open Jupyter from the repository root and execute the notebooks manually in the
listed order. Each experiment notebook defines `FRESH_TRAINING = True` near the
top. Change it to `False` only when resuming an interrupted experiment from its
last checkpoint.

Checkpoints, numerical results, figures, and executed notebooks are kept
together under `training_results/`.

Every experiment saves `training_curves.png`, `evaluation_comparison.png`, and
`qualitative_unseen.png` alongside CSV/JSON metrics. Executed notebook copies
are retained under `training_results/executed_notebooks/`.

With `FRESH_TRAINING = False`, completed experiments reuse their saved results
and interrupted experiments resume from their last completed epoch.

## Final model

After all five experiments finish, the comparison notebook copies the model
with the highest `validation_seen` IoU to:

```text
training_results/best_model.pt
```

`model_registry.json` records all experiment checkpoints so the UI can load and
compare them without retraining.

The UI can load the selected checkpoint and predict at the original image size:

```python
from final_model.inference import load_predictor

predictor = load_predictor("training_results/best_model.pt")
probability = predictor.predict(rgb_uint8_chw, parent_mask_hw, "wheel")
prediction = probability >= 0.5
```

Training requires a local CUDA-capable GPU. Gradient accumulation retains the
controlled effective batch size while reducing GPU memory usage.
