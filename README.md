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
├── deployment/                  # API configuration, Dockerfile, runtime dependencies
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
not retrain or run a model. The user-upload panel can preview files in this
mode, but its Run button needs the inference server described below.

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory web
```

Open <http://127.0.0.1:8000/> and stop the server with `Ctrl+C`.

## Run the website with user-image inference

The upload flow supports:

- an RGB image;
- automatic category and parent-mask prediction with COCO Mask R-CNN;
- selection between multiple detected objects;
- mouse/touch correction with Add, Erase, Undo, and Reset;
- manual upload of a same-size white-on-black parent mask as a fallback;
- a manually confirmed object category;
- a part supported for that category;
- a searchable choice among the 80 COCO parent categories and the trained
  part-query vocabulary for qualitative open-vocabulary testing;
- one of the five final-study models.

Start the combined API and website from the repository root:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn inference_server:app --reload --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>. The first parent prediction can take longer while
the Torchvision Mask R-CNN weights are downloaded and loaded. The first part
prediction similarly loads DINOv2 and OpenCLIP; later requests reuse the loaded
models. Stop the server with `Ctrl+C`.

The API provides:

```text
GET  /api/health          readiness and available model IDs
POST /api/predict         current image + parent-mask part inference
POST /api/parent/predict  automatic category and parent-mask prediction
```

For a separately hosted static frontend, set the
`part-segmentation-api` meta tag in `web/index.html` to the public Python API
origin. Set `PART_DEMO_ALLOWED_ORIGINS` on the API host to the frontend origin,
for example `https://object-relative-part-demo.pages.dev`. Cloudflare Pages can
host the static `web/` directory, but the Python/PyTorch API needs a separate
CPU or GPU service.

The parent detector is an independent COCO-pretrained Mask R-CNN baseline. It
maps supported COCO labels to Pascal-Part-116 parent categories; it does not
alter or retrain the project part-segmentation model. Manual category selection,
mask editing, and mask upload remain available when the automatic result is not
correct. Non-Pascal COCO detections and new combinations of supported object
and part names are shown as experimental open-vocabulary results rather than
evaluated Pascal-Part-116 results. Their dropdowns use curated semantic groups:
for example, `giraffe` offers transferable trained animal parts such as `neck`,
`leg`, and `hoof`, but not unrelated labels such as `door`. The shared mapping
lives in `web/data/inference_options.json` and is validated by both the browser
and API.

## Production API container

The static `web/` directory and the PyTorch API can be served together from one
container. Build from the repository root:

```bash
docker build -f deployment/Dockerfile -t part-segmentation-demo .
docker run --rm -p 8000:8000 part-segmentation-demo
```

The container intentionally runs one Uvicorn worker because every worker would
load another copy of Mask R-CNN, DINOv2, and OpenCLIP. Use a host with enough
memory and persistent model caches, or add the compatible pretrained encoder
files described in Setup to the deployment image. The small trained
segmentation heads under `models/final_study/` are already included. Do not use
`--reload` in production.

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
