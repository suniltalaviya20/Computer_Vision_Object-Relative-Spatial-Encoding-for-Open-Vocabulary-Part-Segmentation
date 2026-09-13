# Final-study local-UI models

These models come from the completed experiments in `training_results/`.

Only compact UI checkpoints are stored here:

- `baseline_object_mask.pt`
- `fixed_uvd.pt`
- `query_gated_uvd.pt`
- `rotation_consistent.pt`
- `geometry_dropout.pt`
- `best_model.pt` (an alias of `rotation_consistent.pt`)

The files are kept directly in this folder without another run-ID directory.
`model_registry.json` is the source of truth for UI names and checkpoint
paths. Training-resume checkpoints, reports,
and executed notebooks remain under `training_results/` and are not duplicated
here.

All models require an RGB image, a binary parent-object mask, and a text part
query. DINOv2 ViT-S/14 and OpenCLIP ViT-B/32 QuickGELU are loaded once and
shared between the five segmentation heads.
