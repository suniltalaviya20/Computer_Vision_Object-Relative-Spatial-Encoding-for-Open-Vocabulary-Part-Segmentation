# Object-Relative Open-Vocabulary Part Segmentation

This project predicts an object-part mask from:

- an RGB image;
- a binary parent-object mask;
- a text query such as `wheel`, `head`, or `wing`.

The final study uses frozen DINOv2 ViT-S/14 visual features, frozen OpenCLIP
ViT-B/32 QuickGELU text features, and one of five trained segmentation heads.

## Current project structure

```text
.
├── dashboard/utils/             # model registry and demo inference
├── data/                        # Pascal-Part-116 (not committed)
├── datasets/                    # dataset and robustness loaders
├── final_training/              # exact final-study architecture and inference
├── models/final_study/          # active deployment checkpoints and registry
├── submission/
│   ├── final_training_notebooks/ # reproducible training notebooks
│   ├── final_training_results/   # metrics, plots, logs, executed notebooks
│   └── trained_points/           # original training/resume artifacts
├── scripts/                     # data, validation, and web export commands
├── src/                         # earlier model and geometry components
└── web/                         # static browser demo
```

The professor submission folder is retained as the complete training record.
The main runtime copies only the smaller `ui_model.pt` artifacts into
`models/final_study/`; `best.pt`, `last.pt`, logs, and reports are not duplicated.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first inference may download DINOv2 and OpenCLIP. Later runs reuse their
local caches. If fully offline, provide the training-compatible files at:

```text
dinov2/
models/pretrained/dinov2_vits14_pretrain.pth
models/pretrained/open_clip/ViT-B-32.pt
```

## Final trained models

The active run is `training_4217799`:

| Model | Selected epoch | Validation IoU |
|---|---:|---:|
| Object-mask baseline | 14 | 0.2814 |
| Fixed UVD | 12 | 0.2855 |
| Query-gated UVD | 12 | 0.2837 |
| Rotation-consistent UVD | 21 | **0.2981** |
| Geometry-dropout UVD | 15 | 0.2871 |

Rotation-consistent UVD is the selected model. Selection used validation IoU;
test metrics were not used for checkpoint selection.

Validate the copied deployment artifacts without loading the large encoders:

```bash
source .venv/bin/activate
python scripts/verify_final_models.py
```

## Dataset

Download and prepare Pascal-Part-116:

```bash
source .venv/bin/activate
python scripts/download_dataset.py
python scripts/prepare_dataset.py
```

Expected locations:

```text
data/raw/PascalPart116/
data/processed/
data/splits/
```

## Run the static website

The website displays predictions exported in advance; selecting an example does
not retrain or run a model.

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory web
```

Open <http://127.0.0.1:8000/> and stop the server with `Ctrl+C`.

## Rebuild website predictions

The registry at `dashboard/utils/demo_registry.py` is the single source of truth
for the five model names and checkpoint paths.

Quickly test real inference first:

```bash
source .venv/bin/activate
python scripts/test_demo_robustness.py
```

Then export the clean catalogue and six robustness conditions:

```bash
python scripts/export_demo_catalogue.py
python scripts/export_demo_robustness.py
```

The export creates the 111-example static demo under:

```text
web/data/catalogue.json
web/assets/examples/
```

## Direct inference

For a new image, supply a `uint8` RGB tensor shaped `[3, H, W]`, a binary
parent mask shaped `[H, W]`, and a text query:

```python
from final_training.inference import load_predictor

predictor = load_predictor(
    "models/final_study/best_model.pt"
)
probability = predictor.predict(rgb_image, parent_mask, "wheel")
prediction = probability >= 0.5
```

This is inference, not training. The parent-object mask is required because U,
V, and D are computed relative to that mask.

## Reproducing training

The full GPU training workflow remains in:

```text
submission/
```

Its notebooks, FAU Slurm script, numerical results, plots, completion markers,
and resume checkpoints are preserved unchanged. They are not required merely to
view the static website or use a deployment checkpoint.
