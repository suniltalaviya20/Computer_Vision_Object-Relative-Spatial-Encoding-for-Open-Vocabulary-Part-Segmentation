# Full-training submission notebooks

This folder contains eight reproducible study notebooks. The five experiment
notebooks optimize models from scratch and therefore require an allocated CUDA
GPU.

Each experiment notebook contains only the material needed for evaluation:
the selected configuration, complete dataset/model/loss/training implementation,
training command, complete epoch history, result tables, figures, and
saved-output locations. The notebooks use the root dataset package and contain
the complete training implementation directly in their cells.
Notebook 06 compares all five models and records the validation-selected
local-UI checkpoint. Notebook 07 then evaluates that selected model using a
separate Pascal-Part-116 benchmark protocol.

## Experimental order

1. `00_data_analysis.ipynb`
2. `01_baseline_object_mask.ipynb`
3. `02_fixed_uvd.ipynb`
4. `03_query_gated_uvd.ipynb`
5. `04_rotation_consistency.ipynb`
6. `05_geometry_branch_dropout.ipynb`
7. `06_final_comparison_and_model_selection.ipynb`
8. `07_pascal_part116_benchmark_comparison.ipynb`

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
listed order. Experiment notebooks 01–05 default to `TRAIN_MODEL = False`. Set it
to `True` to train, keep `FRESH_TRAINING = True` for a new run, or set
`FRESH_TRAINING = False` when resuming an interrupted run.

Checkpoints, numerical results, figures, completion markers, and other generated
reports are kept together under `training_results_corrected/`.

Every experiment saves `training_curves.png`, `evaluation_comparison.png`, and
`qualitative_unseen.png` alongside CSV/JSON metrics.

With `FRESH_TRAINING = False`, completed experiments reuse their saved results
and interrupted experiments resume from their last completed epoch.

Notebook 04 performs two model passes per training batch for its rotation-
consistency objective. For its sampled 90-degree rotations, U and V are swapped
or inverted as required and D is rotated directly on the GPU. This is
mathematically equivalent to recomputing the rotated object-relative geometry,
while avoiding the previous GPU-to-CPU transfers and CPU distance transforms.

## Final model

After all five experiments finish, the comparison notebook copies the model
with the highest `validation_seen` IoU to:

```text
training_results_corrected/best_model.pt
```

`model_registry.json` records all experiment checkpoints so the UI can load and
compare them without retraining.

After notebook 06 has selected the final checkpoint, notebook 07 separately
reports the controlled internal ablation and an Oracle-Obj-like Pascal-Part-116
semantic-class comparison. It audits the training input protocol, reproduces the
official 74-seen/42-unseen class partition, and keeps literature references
separate from non-equivalent internal query-level metrics.

The notebook 07 result is an approximate parent-mask-conditioned,
Oracle-Obj-like comparison, not a strict Oracle-Obj or Pred-All leaderboard
result. Its pipeline audit recorded 76 parent-mask repair activations; the saved
protocol and repair audits under `training_results_corrected/benchmark_comparison/`
must accompany interpretation of the reported benchmark numbers.

The UI can load the selected checkpoint and predict at the original image size:

```python
from final_model.inference import load_predictor

predictor = load_predictor("models/final_study/best_model.pt")
probability = predictor.predict(rgb_uint8_chw, parent_mask_hw, "wheel")
prediction = probability >= 0.5
```

Training requires a local CUDA-capable GPU. Gradient accumulation retains the
controlled effective batch size while reducing GPU memory usage.
