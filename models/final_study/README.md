# Final-study deployment models

These models come from the completed run `training_4217799` in
`submission/trained_points/training_4217799`.

Only deployment checkpoints are stored here:

- `baseline_object_mask.pt`
- `fixed_uvd.pt`
- `query_gated_uvd.pt`
- `rotation_consistent.pt`
- `geometry_dropout.pt`
- `best_model.pt` (an alias of `rotation_consistent.pt`)

The run ID is recorded in `model_registry.json`; because this is the only active
final run, its files are kept directly in this folder without another run-ID
directory. The dashboard registry in `final_model/demo_registry.py` is the source of
truth for UI names and checkpoint paths. Training-resume checkpoints, reports,
logs, and executed notebooks remain in the professor submission folder and are
not duplicated here.

All models require an RGB image, a binary parent-object mask, and a text part
query. DINOv2 ViT-S/14 and OpenCLIP ViT-B/32 QuickGELU are loaded once and
shared between the five segmentation heads.
