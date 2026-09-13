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
├── data/                        # Pascal-Part-116 (not committed)
├── datasets/                    # Pascal-Part loader and label metadata
├── deployment/                  # local FastAPI server
├── final_model/                 # final architecture, training, and inference
├── models/final_study/          # active deployment checkpoints and registry
├── final_training_notebooks/    # reproducible training notebooks
├── training_results/            # metrics, plots and training artifacts
├── scripts/                     # notebook training workflow
├── tests/                       # layout and API regression checks
├── requirements.txt             # project dependencies
├── README.md
└── web/                         # static browser demo
```

The notebooks and training results are retained as the complete training record.
The main runtime copies only the smaller `ui_model.pt` artifacts into
`models/final_study/`; `best.pt`, `last.pt`, logs, and reports are not duplicated.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first training or inference run downloads DINOv2 and OpenCLIP from their
official online sources. Later runs reuse the standard local caches.

## Final trained models

| Model | Selected epoch | Validation IoU |
|---|---:|---:|
| Object-mask baseline | 14 | 0.2814 |
| Fixed UVD | 12 | 0.2855 |
| Query-gated UVD | 12 | 0.2837 |
| Rotation-consistent UVD | 21 | **0.2981** |
| Geometry-dropout UVD | 15 | 0.2871 |

Rotation-consistent UVD is the selected model. Selection used validation IoU;
test metrics were not used for checkpoint selection.

## Dataset

Download and prepare Pascal-Part-116 from the repository root:

```bash
source .venv/bin/activate
python scripts/download_dataset.py
python scripts/prepare_dataset.py
```

The scripts create and validate these locations:

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

### Local quick start

Open a terminal in the project root. Create the virtual environment only if it
does not already exist, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On later runs, activate the existing environment and start the application:

```bash
source .venv/bin/activate
uvicorn deployment.inference_server:app --reload --host 127.0.0.1 --port 8000
```

Keep that terminal open. Wait until it prints:

```text
Uvicorn running on http://127.0.0.1:8000
```

Then open <http://127.0.0.1:8000/> in the browser. Use `http`, not `https`.
If an older page is cached, press `Ctrl+Shift+R`. Stop the application by
pressing `Ctrl+C` in the terminal.

Do not start the separate `python -m http.server` command at the same time on
port 8000. The Uvicorn application serves both the webpage and inference API.

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

The first parent prediction can take longer while the Torchvision Mask R-CNN
weights are downloaded and loaded. The first part prediction similarly loads
DINOv2 and OpenCLIP; later requests reuse the loaded models.

The API provides:

```text
GET  /api/health          readiness and available model IDs
POST /api/predict         current image + parent-mask part inference
POST /api/parent/predict  automatic category and parent-mask prediction
```

Serve the frontend and API together with Uvicorn so the browser can reach the
same-origin `/api/` endpoints.

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

The prepared 111-example static demo is stored under:

```text
web/data/catalogue.json
web/assets/examples/
```

## Direct inference

For a new image, supply a `uint8` RGB tensor shaped `[3, H, W]`, a binary
parent mask shaped `[H, W]`, and a text query:

```python
from final_model.inference import load_predictor

predictor = load_predictor(
    "models/final_study/best_model.pt"
)
probability = predictor.predict(rgb_image, parent_mask, "wheel")
prediction = probability >= 0.5
```

This is inference, not training. The parent-object mask is required because U,
V, and D are computed relative to that mask.

## Reproducing training

The source notebooks are in `final_training_notebooks/`. Existing metrics,
plots, executed notebooks, inference files, and resume checkpoints are
together in `training_results/`.

Open Jupyter from the repository root and run these notebooks manually in order:

```text
00_data_analysis.ipynb
01_baseline_object_mask.ipynb
02_fixed_uvd.ipynb
03_query_gated_uvd.ipynb
04_rotation_consistency.ipynb
05_geometry_branch_dropout.ipynb
06_final_comparison_and_model_selection.ipynb
```

Each experiment notebook has a visible `FRESH_TRAINING` setting. Keep it `True`
for a new run. Change it to `False` before restarting an interrupted notebook so
training resumes from its last saved epoch.

The numerical results, plots, completion markers, and resume checkpoints are
kept under `training_results/`. They are not required merely to view the static
website or use an inference checkpoint.

## Regression checks

Run the layout and HTTP API tests without downloading encoder weights:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

The API tests use controlled model outputs to check uploads, model selection,
parent detection and static assets. Real predictions require the encoder weights.
