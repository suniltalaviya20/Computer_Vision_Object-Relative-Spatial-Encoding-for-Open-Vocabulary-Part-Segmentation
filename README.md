# Computer Vision: Object-Relative Spatial Encoding for Open-Vocabulary Part Segmentation

This project uses Pascal-Part-116, containing RGB images, parent-object annotations, and object-part annotations. Dataset files are not stored in Git.

## Setup

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download the Dataset

```bash
python scripts/download_dataset.py
```

The dataset is extracted to:

```text
data/raw/PascalPart116/
```

## Prepare the Dataset

```bash
python scripts/prepare_dataset.py
```

This validates the images and annotations, splits the data by image ID, and creates one record for every annotated object part. The original dataset is not modified.

Generated manifests and split files are stored in:

```text
data/processed/
data/splits/
```

Existing valid outputs are reused. To regenerate them:

```bash
python scripts/prepare_dataset.py --force
```

## Inspect the Dataset

Display a random prepared sample:

```bash
python scripts/inspect_dataset.py --split train
```

Available splits are `train`, `validation`, `test`, `train_seen`, `validation_seen`, `test_seen`, and `test_unseen`.

Visualizations are saved to:

```text
outputs/dataset_inspection/
```

## Dataset Analysis

```bash
jupyter notebook notebooks/data_analysis.ipynb
```

The notebook can also be opened in VS Code using the project `.venv` as its kernel. Generated plots are saved to `outputs/data_analysis/`.

The current dataset loader preserves the original image sizes and annotations. Resizing, normalization, geometry, and model training will be added later.
