# Computer Vision: Object-Relative Spatial Encoding for Open-Vocabulary Part Segmentation

This project studies text-conditioned object-part segmentation using RGB appearance, a parent-object mask, a text query, and optional object-relative spatial encoding.

Given:
- an RGB image,
- a parent-object mask,
- a text query such as `head`, `wheel`, or `ear`,

the model predicts a binary mask for the requested part.

The project uses Pascal-Part-116. Dataset files and trained model checkpoints are not stored directly in Git.

---

## 1. Clone the Repository

```bash
git clone <repository-url>
cd Computer_Vision_Object-Relative-Spatial-Encoding-for-Open-Vocabulary-Part-Segmentation
```

---

## 2. Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

# Dataset

## 3. Download Pascal-Part-116

```bash
python scripts/download_dataset.py
```

The raw dataset should be located at:

```text
data/
└── raw/
    └── PascalPart116/
```

The raw dataset is ignored by Git.

## 4. Prepare the Dataset

```bash
python scripts/prepare_dataset.py
```

Generated files are stored under:

```text
data/
├── processed/
└── splits/
```

To force regeneration:

```bash
python scripts/prepare_dataset.py --force
```

The project uses:

```text
train
validation
test
train_seen
validation_seen
test_seen
test_unseen
```

---

# Run the Existing Static Web Demo

The repository contains a pre-exported static demo in:

```text
web/
├── index.html
├── styles.css
├── app.js
├── data/
│   └── catalogue.json
└── assets/
    └── examples/
```

Start it with:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory web
```

Open:

```text
http://127.0.0.1:8000/
```

If `web/data/catalogue.json` and `web/assets/` are already present, the dataset and model checkpoints are not required just to view the website.

---

# Model Checkpoints

## 5. Where Checkpoints Go

Model checkpoints live under:

```text
outputs/object_zoom/
```

Example:

```text
outputs/
└── object_zoom/
    ├── alignment_mask/
    │   └── best.pt
    ├── alignment_fixed_uvd/
    │   └── best.pt
    ├── alignment_query_gated_uvd/
    │   └── best.pt
    └── alignment_relative_uv/
        └── best.pt
```

The `outputs/` directory is ignored by Git, so `.pt` files must be copied or downloaded separately.

---

# Using Your Own `.pt` Model

This is the recommended workflow for testing a colleague's own checkpoint.

## 6. Put the Checkpoint in Its Own Folder

Do not overwrite an existing checkpoint if possible.

Use a unique path such as:

```text
outputs/object_zoom/my_fixed_uvd_v1/best.pt
```

A unique checkpoint path is important because exported results are associated with the registered checkpoint path. Reusing the exact old path can make old exported results look current.

---

## 7. Register the Model

Open:

```text
dashboard/utils/demo_registry.py
```

This file is the source of truth for the demo models.

Add a new registry entry, or replace an existing entry if you intentionally want your checkpoint to replace one of the current demo models.

Example:

```python
{
    "id": "my_fixed_uvd_v1",
    "label": "My Fixed UVD v1",

    # Must match how this checkpoint was trained.
    "mode": "alignment_fixed_uvd",

    "checkpoint": (
        PROJECT_ROOT
        / "outputs"
        / "object_zoom"
        / "my_fixed_uvd_v1"
        / "best.pt"
    ),

    "geometry": ["u", "v", "d"],
    "gated": False,
    "comparison_group": "final_object_centric",
    "supports_live": True,
}
```

Use a unique `id`.

The `id` is used in:
- `catalogue.json`,
- exported asset folders,
- frontend model selection,
- robustness results.

---

## 8. Match the Registry to the Model Type

### Parent-mask / crop + alignment model

```python
"mode": "alignment_mask",
"geometry": [],
"gated": False,
```

### Relative U/V model

```python
"mode": "alignment_relative_uv",
"geometry": ["u", "v"],
"gated": False,
```

### Fixed U/V/D model

```python
"mode": "alignment_fixed_uvd",
"geometry": ["u", "v", "d"],
"gated": False,
```

### Query-gated U/V/D model

```python
"mode": "alignment_query_gated_uvd",
"geometry": ["u", "v", "d"],
"gated": True,
```

`D` is the 2D distance to the parent-mask boundary. It is not depth.

---

## 9. Architecture Compatibility

Changing only the `.pt` path is enough only when the checkpoint uses an architecture already supported by this project.

The loader is:

```text
dashboard/utils/models.py
```

The model implementations are under:

```text
src/
```

If your checkpoint was trained with one of the existing supported modes, register the correct mode and path.

If your checkpoint uses:
- a new architecture,
- a different decoder,
- different tensor shapes,
- a new geometry representation,
- a different set of saved state-dict keys,

then `dashboard/utils/models.py` and/or the relevant code in `src/` must also be updated before the checkpoint can load.

A `.pt` file is not automatically compatible just because it exists.

---

# Rebuild the Demo After Changing Models

## 10. Test the Registered Models

Run:

```bash
python scripts/test_demo_robustness.py
```

The script should finish with:

```text
ROBUSTNESS TEST COMPLETE
```

If loading fails, fix the registry/model configuration before exporting website data.

---

## 11. Regenerate the Clean Demo Catalogue

Whenever the registered models change, run:

```bash
python scripts/export_demo_catalogue.py
```

This regenerates the clean examples and synchronizes:

```text
web/data/catalogue.json
```

with the current registry.

---

## 12. Regenerate Robustness Results

After the clean catalogue is regenerated:

```bash
python scripts/export_demo_robustness.py
```

This generates:

```text
Rotate 15°
Rotate 45°
Rotate 90°
Erode parent mask
Dilate parent mask
Shift parent mask
```

The robustness exporter contains:

```python
ROBUSTNESS_PIPELINE_VERSION = 2
```

Increase that version only when the robustness preprocessing/export pipeline changes.

For a new model checkpoint, prefer a new checkpoint path and a new model `id`.

---

# Replacing an Existing Demo Model

To make your model take the place of an existing one:

1. Put your checkpoint at a unique path, for example:

```text
outputs/object_zoom/my_model_v1/best.pt
```

2. Open:

```text
dashboard/utils/demo_registry.py
```

3. Replace the registry entry you no longer want.

4. Give your model a unique `id`.

5. Set the correct:

```text
mode
geometry
gated
checkpoint
```

6. Test:

```bash
python scripts/test_demo_robustness.py
```

7. Re-export clean results:

```bash
python scripts/export_demo_catalogue.py
```

8. Re-export robustness results:

```bash
python scripts/export_demo_robustness.py
```

9. Start or refresh the website:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory web
```

10. Hard refresh the browser if needed:

```text
Ctrl + Shift + R
```

The frontend reads the model list from the exported catalogue, so model names should not need to be hard-coded in `web/app.js`.

---

# Adding a New Model Without Removing Existing Models

1. Add a new unique registry entry in:

```text
dashboard/utils/demo_registry.py
```

2. Put the checkpoint at:

```text
outputs/object_zoom/<your-model-folder>/best.pt
```

3. Run:

```bash
python scripts/test_demo_robustness.py
```

4. Re-export:

```bash
python scripts/export_demo_catalogue.py
python scripts/export_demo_robustness.py
```

5. Refresh the website.

The model should then appear as another option if the current frontend/export pipeline supports the number of registered models.

---

# Demo Registry

Treat:

```text
dashboard/utils/demo_registry.py
```

as the single source of truth for:
- model ID,
- display label,
- inference mode,
- checkpoint path,
- geometry inputs,
- whether the model is gated,
- whether it is available to the demo.

Avoid hard-coding checkpoint paths separately in exporters or frontend code.

---

# Important Demo Files

```text
dashboard/utils/demo_registry.py
```

Defines registered demo models.

```text
dashboard/utils/models.py
```

Loads datasets, DINOv2, CLIP, checkpoints, and runs predictions.

```text
datasets/object_centric_dataset.py
```

Builds normal object-centric crops and geometry.

```text
datasets/object_centric_robustness_dataset.py
```

Applies robustness perturbations and rebuilds object-centric inputs.

```text
src/
```

Contains model, geometry, crop, projection, preprocessing, robustness, DINOv2, and CLIP code.

```text
scripts/export_demo_catalogue.py
```

Exports clean demo examples.

```text
scripts/export_demo_robustness.py
```

Exports robustness examples.

```text
scripts/test_demo_robustness.py
```

Runs a quick real inference check before a large export.

---

# Demo Data Flow

```text
Pascal-Part-116
      ↓
prepared split
      ↓
object-centric dataset
      ↓
RGB crop + parent mask
      ↓
relative U / V / D when required
      ↓
DINOv2 visual encoder
      +
CLIP text encoder
      ↓
registered segmentation model
      ↓
crop prediction
      ↓
projection to full image
      ↓
IoU / Dice / leakage
      ↓
web/data/catalogue.json
+
web/assets/examples/
```

For robustness conditions, the raw image and/or parent mask is perturbed before the normal object-centric crop and geometry are recomputed.

---

# Inspect the Demo Dataset

```bash
python scripts/inspect_demo_dataset.py
```

---

# Original Dataset Inspection

```bash
python scripts/inspect_dataset.py --split train
```

---

# Dataset Analysis

```bash
jupyter notebook notebooks/data_analysis.ipynb
```

---

# Main Project Structure

```text
.
├── dashboard/
│   └── utils/
│       ├── demo_registry.py
│       └── models.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── datasets/
├── notebooks/
├── outputs/
│   └── object_zoom/
│       └── <model-folder>/
│           └── best.pt
├── scripts/
├── src/
├── web/
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   ├── data/
│   │   └── catalogue.json
│   └── assets/
├── README.md
└── requirements.txt
```

---

# Quick Workflow for a Colleague's Model

```text
1. Clone the repository
2. Install requirements
3. Download and prepare Pascal-Part-116
4. Put the checkpoint at:
   outputs/object_zoom/<unique-model-name>/best.pt
5. Add the model to:
   dashboard/utils/demo_registry.py
6. Make sure mode / geometry / gated match the checkpoint
7. Run:
   python scripts/test_demo_robustness.py
8. Run:
   python scripts/export_demo_catalogue.py
9. Run:
   python scripts/export_demo_robustness.py
10. Run:
    python -m http.server 8000 --bind 127.0.0.1 --directory web
11. Open:
    http://127.0.0.1:8000/
```

If the checkpoint uses a new architecture instead of one of the existing supported modes, update the Python model-loading/model-definition code before steps 7-9.