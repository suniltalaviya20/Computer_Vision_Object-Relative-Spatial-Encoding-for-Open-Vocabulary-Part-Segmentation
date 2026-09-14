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
├── data/                        # tracked splits; downloaded/prepared data are ignored
├── datasets/                    # Pascal-Part loader and label metadata
├── deployment/                  # local FastAPI server
├── final_model/                 # final architecture, training, and inference
├── models/final_study/          # active deployment checkpoints and registry
├── final_training_notebooks/    # reproducible training notebooks
├── training_results_corrected/  # corrected metrics, plots and training artifacts
├── scripts/                     # dataset preparation and static-web export
├── tests/                       # layout and API regression checks
├── requirements.txt             # project dependencies
├── README.md
└── web/                         # static browser demo
```

The source notebooks and generated training results form the training record.
The deployment directory contains only compact inference-checkpoint copies;
training `best.pt`/`last.pt` checkpoints, logs, and reports remain under
`training_results_corrected/` and are not duplicated.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use Python 3.10 or newer. Training requires a CUDA-capable GPU; direct inference
and the local API can also run on CPU, although they will be slower. The first
training or part-inference run requires internet access to download DINOv2 and
OpenCLIP. Automatic parent-object detection separately downloads Torchvision's
COCO Mask R-CNN weights. Later runs reuse the standard local caches.

## Final trained models

| Model | Selected epoch | Validation IoU | Seen test IoU | Unseen test IoU |
|---|---:|---:|---:|---:|
| Object-mask baseline | 20 | 0.2880 | 0.2956 | 0.2243 |
| Fixed UVD | 15 | 0.2912 | 0.3015 | 0.2493 |
| Query-gated UVD | 15 | 0.2909 | **0.3028** | 0.2456 |
| Rotation-consistent UVD | 14 | **0.2937** | 0.3028 | **0.2712** |
| Geometry-dropout UVD | 17 | 0.2878 | 0.2972 | 0.2415 |

Rotation-consistent UVD is the selected model. Selection used validation IoU;
test metrics were not used for checkpoint selection.

### Pascal-Part-116 benchmark context

Notebook 07 also evaluates the selected model with multiclass decoding inside
ground-truth parent-object regions. It reports 47.78% seen mIoU, 31.31% unseen
mIoU, and 37.83% harmonic IoU over the classes with valid test support.

This is an approximate parent-mask-conditioned, Oracle-Obj-like comparison—not
a strict Oracle-Obj or Pred-All leaderboard result. The model was trained as
independent binary part queries at 224 px, while the official evaluator predicts
a multiclass part map per oracle object region. The pipeline audit also recorded
76 parent-mask repair activations, so the benchmark should be interpreted with
the protocol qualifications saved in
`training_results_corrected/benchmark_comparison/`.

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

The raw and processed datasets are intentionally excluded from version control;
the deterministic split files under `data/splits/` are tracked.

## Run the web demo

After completing the project setup above, start the application from the
repository root:

```bash
source .venv/bin/activate
uvicorn deployment.inference_server:app --reload --host 127.0.0.1 --port 8000
```

Keep the terminal open and wait until it prints:

```text
Uvicorn running on http://127.0.0.1:8000
```

Then open <http://127.0.0.1:8000/> in the browser. Use `http`, not `https`.
If an older page is cached, press `Ctrl+Shift+R`. Stop the application by
pressing `Ctrl+C` in the terminal.

The Uvicorn application serves both the webpage and inference API.

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

Regenerate it from the deployed checkpoints after running notebook 06:

```bash
source .venv/bin/activate
python scripts/export_web_assets.py --publish
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
plots, compact inference checkpoints, completion markers, and training-resume
checkpoints are stored under `training_results_corrected/`.

Open Jupyter from the repository root and run these notebooks manually in order:

```text
00_data_analysis.ipynb
01_baseline_object_mask.ipynb
02_fixed_uvd.ipynb
03_query_gated_uvd.ipynb
04_rotation_consistency.ipynb
05_geometry_branch_dropout.ipynb
06_final_comparison_and_model_selection.ipynb
07_pascal_part116_benchmark_comparison.ipynb
```

Experiment notebooks 01–05 default to `TRAIN_MODEL = False`, so running them
loads the saved tables and plots without training. Set `TRAIN_MODEL = True` to
train; keep `FRESH_TRAINING = True` for a new run, or set it to `False` to resume
an interrupted run.

Notebook 04 performs two model passes per training batch for its rotation-
consistency objective. Its rotated UVD maps are transformed directly on the GPU
for the sampled 90-degree rotation, avoiding CPU distance-transform
recomputation without changing the experiment definition.

The numerical results, plots, completion markers, and resume checkpoints are
kept under `training_results_corrected/`. They are not required merely to view
the static website or use an inference checkpoint.

## Regression checks

Run the layout and HTTP API tests without downloading encoder weights:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

The API tests use controlled model outputs to check uploads, model selection,
parent detection and static assets. Real predictions require the encoder weights.

## References and attribution

- [OV-PARTS protocol and Pascal-Part-116 resources](https://github.com/OpenRobotLab/OV_PARTS)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)

The downloaded datasets and pretrained model weights remain subject to their
respective upstream licenses and terms.
