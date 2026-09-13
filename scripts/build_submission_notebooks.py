#!/usr/bin/env python3
"""Embed the exact training implementation into the submission notebooks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "final_training_notebooks"
EXECUTED_DIR = (
    ROOT
    / "training_results"
    / "executed_notebooks"
)

EXPERIMENTS = {
    "01_baseline_object_mask.ipynb": (
        "baseline_object_mask",
        "01 — Object-mask baseline",
        "Reference model using image, text, and the parent-object mask without active UVD geometry.",
    ),
    "02_fixed_uvd.ipynb": (
        "fixed_uvd",
        "02 — Fixed object-relative UVD",
        "Adds normalized horizontal U, vertical V, and boundary-distance D with fixed gates.",
    ),
    "03_query_gated_uvd.ipynb": (
        "query_gated_uvd",
        "03 — Query-gated object-relative UVD",
        "Uses the text query to learn separate sigmoid gates for U, V, and D.",
    ),
    "04_rotation_consistency.ipynb": (
        "rotation_consistent",
        "04 — Rotation-consistent query-gated UVD",
        "Adds 90-degree rotation augmentation and prediction-consistency loss.",
    ),
    "05_geometry_branch_dropout.ipynb": (
        "geometry_dropout",
        "05 — Geometry-branch dropout",
        "Drops the complete UVD branch for 30% of training examples.",
    ),
}

SETUP = '''from pathlib import Path
import json
import os
import sys

from IPython.display import Image, display

PROJECT_ROOT = next(
    (path for path in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
     if (path / "datasets").is_dir() and (path / "final_training").is_dir()),
    None,
)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Run from the repository or final_training_notebooks directory")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUN_ID = os.environ.get("FINAL_TRAINING_RUN_ID", "manual")
RESULT_ROOT = PROJECT_ROOT / "training_results"
POINT_ROOT = RESULT_ROOT
print("Project:", PROJECT_ROOT)
print("Run ID:", RUN_ID)'''


def markdown(source: str, cell_id: str | None = None) -> dict:
    cell = {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}
    if cell_id:
        cell["id"] = cell_id
    return cell


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def notebook(cells: list[dict], metadata: dict) -> dict:
    for index, cell in enumerate(cells):
        cell.setdefault("id", f"cell-{index:02d}")
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


def between(source: str, start: str, end: str | None = None) -> str:
    start_index = source.index(start)
    end_index = source.index(end, start_index) if end else len(source)
    return source[start_index:end_index].strip()


def embedded_training_sections() -> tuple[str, str, str, str]:
    source = (ROOT / "final_training" / "training_core.py").read_text()
    source = source.replace("from __future__ import annotations\n\n", "", 1)
    project_block = '''PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

'''
    source = source.replace(project_block, "", 1)
    model_marker = "class PartSegmenter(nn.Module):"
    runtime_marker = "class TrainingRuntime:"
    run_marker = "def run_experiment(experiment: str) -> pd.DataFrame:"
    data_marker = "def run_data_analysis() -> pd.DataFrame:"
    return (
        source[: source.index(model_marker)].strip(),
        between(source, model_marker, runtime_marker),
        between(source, runtime_marker, run_marker),
        between(source, run_marker, data_marker),
    )


def training_notebook(
    experiment: str,
    title: str,
    description: str,
    metadata: dict,
) -> dict:
    preparation, model, runtime, runner = embedded_training_sections()
    rotation_note = ""
    if experiment == "rotation_consistent":
        rotation_note = (
            "\n\n**Implementation note:** RGB, masks, U, V, and D are spatially co-rotated; "
            "U and V are not recomputed in the rotated image frame."
        )
    cells = [
        markdown(f"# {title}\n\n{description}{rotation_note}"),
        code(SETUP),
        markdown("## Configuration, preprocessing, and dataset"),
        code(preparation),
        markdown("## Model and loss"),
        code(model),
        markdown("## Training, evaluation, and checkpoint logic"),
        code(runtime),
        code(runner),
        markdown("## Experiment configuration"),
        code(
            f'''EXPERIMENT = "{experiment}"
CONFIG = TrainingConfig(experiment=EXPERIMENT)
RESULT_DIR = RESULT_ROOT / EXPERIMENT
CHECKPOINT_DIR = POINT_ROOT / EXPERIMENT
display(pd.Series({{**asdict(CONFIG), **EXPERIMENTS[EXPERIMENT]}}, name="value").to_frame())'''
        ),
        markdown("## Train and save"),
        code(
            '''if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for training")
print("GPU:", torch.cuda.get_device_name(0))
summary = run_experiment(EXPERIMENT)
display(summary.round(4))'''
        ),
        markdown("## Epochs, results, and figures"),
        code(
            '''history = pd.read_csv(RESULT_DIR / "history.csv")
completion = json.loads((RESULT_DIR / "completion.json").read_text())
summary = pd.read_csv(RESULT_DIR / "summary.csv")
display(history.round(4))
display(summary.round(4))
print(
    f"Stopped: {completion['stop_reason']} | epochs: {completion['epochs_completed']} | "
    f"selected epoch: {completion['selected_epoch']} | "
    f"validation IoU: {completion['best_validation_iou']:.4f}"
)
display(Image(filename=str(RESULT_DIR / "training_curves.png")))
display(Image(filename=str(RESULT_DIR / "evaluation_comparison.png")))
display(Image(filename=str(RESULT_DIR / "qualitative_unseen.png")))'''
        ),
        markdown(
            '''## Saved outputs

`last.pt`, `best.pt`, `ui_model.pt`, complete epoch history, detailed predictions, summary metrics,
figures, configuration, and completion metadata are saved automatically.'''
        ),
    ]
    return notebook(cells, metadata)


def data_analysis_code() -> str:
    source = (ROOT / "final_training" / "training_core.py").read_text()
    function = between(source, "def run_data_analysis()", "def build_comparison()")
    return '''import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from datasets import PascalPart116Dataset

def run_id():
    return os.environ.get("FINAL_TRAINING_RUN_ID", "manual")

def results_root():
    path = PROJECT_ROOT / "training_results"
    path.mkdir(parents=True, exist_ok=True)
    return path

''' + function


def data_notebook(metadata: dict) -> dict:
    return notebook(
        [
            markdown("# 00 — Dataset analysis\n\nAudits all prepared Pascal-Part-116 splits."),
            code(SETUP),
            markdown("## Dataset-analysis implementation"),
            code(data_analysis_code()),
            markdown("## Run analysis and display saved graph"),
            code(
                '''RESULT_DIR = RESULT_ROOT / "data_analysis"
overview = run_data_analysis()
display(overview)
display(Image(filename=str(RESULT_DIR / "dataset_overview.png")))'''
            ),
        ],
        metadata,
    )


def comparison_code() -> str:
    source = (ROOT / "final_training" / "training_core.py").read_text()
    function = between(source, "def build_comparison()")
    names = [experiment for experiment, _, _ in EXPERIMENTS.values()]
    return f'''import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

EXPERIMENTS = {dict.fromkeys(names)!r}

def run_id():
    return os.environ.get("FINAL_TRAINING_RUN_ID", "manual")

def results_root():
    path = PROJECT_ROOT / "training_results"
    path.mkdir(parents=True, exist_ok=True)
    return path

def points_root():
    path = PROJECT_ROOT / "training_results"
    path.mkdir(parents=True, exist_ok=True)
    return path

{function}'''


def comparison_notebook(metadata: dict) -> dict:
    return notebook(
        [
            markdown(
                "# 06 — Final model comparison\n\n"
                "Selects the local-UI model using validation-seen IoU only."
            ),
            code(SETUP),
            markdown("## Comparison and selection implementation"),
            code(comparison_code()),
            markdown("## Run comparison"),
            code(
                '''comparison = build_comparison()
display(comparison.round(4))
display(Image(filename=str(RESULT_ROOT / "final_model_comparison.png")))
registry = json.loads((POINT_ROOT / "model_registry.json").read_text())
display(pd.DataFrame(registry["models"]).T.sort_values("validation_iou", ascending=False).round(4))
print("Selected model:", registry["selected_model"])
print("Selected UI checkpoint:", POINT_ROOT / "best_model.pt")'''
            ),
        ],
        metadata,
    )


def clean_executed_notebooks() -> None:
    for path in sorted(EXECUTED_DIR.glob("*.ipynb")):
        payload = json.loads(path.read_text())
        payload["cells"] = [
            cell
            for cell in payload["cells"]
            if not str(cell.get("id", "")).startswith("submission-")
        ]
        if path.name == "00_data_analysis.ipynb":
            summary = "## Saved graph\n\n![Dataset overview](../data_analysis/dataset_overview.png)"
        elif path.name == "06_final_comparison_and_model_selection.ipynb":
            summary = "## Saved comparison\n\n![Final model comparison](../final_model_comparison.png)"
        else:
            experiment = EXPERIMENTS[path.name][0]
            summary = f'''## Saved figures

![Training curves](../{experiment}/training_curves.png)

![Seen and unseen metrics](../{experiment}/evaluation_comparison.png)

![Qualitative unseen predictions](../{experiment}/qualitative_unseen.png)'''
        payload["cells"].append(markdown(summary, "submission-summary"))
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")


def main() -> None:
    metadata = {
        path.name: json.loads(path.read_text()).get("metadata", {})
        for path in NOTEBOOK_DIR.glob("*.ipynb")
    }
    outputs = {"00_data_analysis.ipynb": data_notebook(metadata["00_data_analysis.ipynb"])}
    for filename, (experiment, title, description) in EXPERIMENTS.items():
        outputs[filename] = training_notebook(experiment, title, description, metadata[filename])
    final_name = "06_final_comparison_and_model_selection.ipynb"
    outputs[final_name] = comparison_notebook(metadata[final_name])
    for filename, payload in outputs.items():
        (NOTEBOOK_DIR / filename).write_text(
            json.dumps(payload, indent=1, ensure_ascii=False) + "\n"
        )
    clean_executed_notebooks()


if __name__ == "__main__":
    main()
