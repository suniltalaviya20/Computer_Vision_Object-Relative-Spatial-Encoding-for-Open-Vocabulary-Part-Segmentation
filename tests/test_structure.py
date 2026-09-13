"""Repository layout checks: standard library only, no encoders or GPU."""

import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DIRS = ("final_model", "datasets", "deployment")


def active_python():
    yield ROOT / "inference_server.py"
    yield ROOT / "inference_options.py"
    for directory in ACTIVE_DIRS:
        yield from (ROOT / directory).rglob("*.py")


class StructureTests(unittest.TestCase):
    def test_layout(self):
        for directory in (*ACTIVE_DIRS, "web", "tests", "final_training_notebooks",
                          "scripts", "data", "models", "training_results"):
            with self.subTest(directory=directory):
                self.assertTrue((ROOT / directory).is_dir())
        for filename in ("__init__.py", "training_core.py", "inference.py"):
            self.assertTrue((ROOT / "final_model" / filename).is_file())
        for filename in ("download_dataset.py", "prepare_dataset.py"):
            self.assertTrue((ROOT / "scripts" / filename).is_file())
        for old in ("src", "dashboard", "tools", "experiments", "deployment/Dockerfile",
                    "deployment/requirements.txt", "deployment/requirements-api.txt",
                    "scripts/run_training_notebooks.sh",
                    "scripts/run_full_training_overnight.sh"):
            self.assertFalse((ROOT / old).exists(), old)
        self.assertFalse(any((ROOT / "final_training").glob("*.py")))

    def test_active_imports(self):
        migrated = set()
        for path in active_python():
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                names = ([node.module or ""] if isinstance(node, ast.ImportFrom)
                         else [alias.name for alias in node.names] if isinstance(node, ast.Import)
                         else [])
                for name in names:
                    self.assertNotIn(name.split(".")[0], ("src", "final_training", "dashboard", "scripts", "submission"), str(path))
                    if name.startswith("final_model."):
                        migrated.add(name)
                        module = ROOT.joinpath(*name.split("."))
                        self.assertTrue(module.with_suffix(".py").is_file() or (module / "__init__.py").is_file())
        self.assertIn("final_model.inference", migrated)

    def test_training_artifacts_present(self):
        result_root = ROOT / "training_results"
        for experiment in (
            "baseline_object_mask", "fixed_uvd", "query_gated_uvd",
            "rotation_consistent", "geometry_dropout",
        ):
            for filename in (
                "best.pt", "last.pt", "ui_model.pt", "summary.csv",
                "history.csv", "training_curves.png",
                "evaluation_comparison.png", "qualitative_unseen.png",
            ):
                with self.subTest(experiment=experiment, filename=filename):
                    self.assertTrue((result_root / experiment / filename).is_file())
        self.assertTrue((result_root / "best_model.pt").is_file())
        self.assertTrue((result_root / "model_registry.json").is_file())

    def test_notebook_paths(self):
        for path in sorted((ROOT / "final_training_notebooks").glob("*.ipynb")):
            notebook = json.loads(path.read_text())
            source = "\n".join(
                "".join(cell.get("source", []))
                for cell in notebook["cells"]
            )
            with self.subTest(notebook=path.name):
                self.assertNotIn('path / "final_training"', source)
                self.assertNotIn("final_training.training_core", source)
                self.assertIn('path / "final_model"', source)
                self.assertIn("FRESH_TRAINING = True", source)

    def test_registry_paths(self):
        registry = json.loads((ROOT / "models/final_study/model_registry.json").read_text())
        for entry in registry["models"].values():
            self.assertTrue((ROOT / entry["checkpoint"]).is_file())
        self.assertTrue((ROOT / registry["selected_checkpoint"]).is_file())

    def test_catalogue_assets(self):
        references = []

        def visit(value):
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
            elif isinstance(value, str) and value.startswith("assets/"):
                references.append(value)
                self.assertTrue((ROOT / "web" / value).is_file(), value)
            elif isinstance(value, str) and value.startswith("models/final_study/"):
                self.assertTrue((ROOT / value).is_file(), value)

        visit(json.loads((ROOT / "web/data/catalogue.json").read_text()))
        self.assertTrue(references)

if __name__ == "__main__":
    unittest.main()
