"""Repository layout checks: standard library only, no encoders or GPU."""

import ast
import importlib.util
import json
from pathlib import Path
import re
import shlex
import unittest


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DIRS = ("final_model", "tools", "src", "datasets", "experiments", "deployment")


def active_python():
    yield ROOT / "inference_server.py"
    for directory in ACTIVE_DIRS:
        yield from (ROOT / directory).rglob("*.py")


class StructureTests(unittest.TestCase):
    def test_layout(self):
        for directory in (*ACTIVE_DIRS, "web", "tests", "submission/final_training",
                          "submission/scripts", "data", "models", "outputs",
                          "experiments/feature_extraction", "experiments/crop_alignment_uvd"):
            with self.subTest(directory=directory):
                self.assertTrue((ROOT / directory).is_dir())
        for filename in ("__init__.py", "training_core.py", "inference.py",
                         "demo_models.py", "demo_registry.py"):
            self.assertTrue((ROOT / "final_model" / filename).is_file())
        for filename in ("download_dataset.py", "prepare_dataset.py", "inspect_dataset.py",
                         "inspect_demo_dataset.py", "verify_final_models.py",
                         "test_demo_robustness.py", "export_demo_catalogue.py",
                         "export_demo_robustness.py", "export_demo_sample.py"):
            self.assertTrue((ROOT / "tools" / filename).is_file())
        for old in ("final_training", "scripts", "dashboard", "deployment/requirements-api.txt"):
            self.assertFalse((ROOT / old).exists(), old)
        self.assertTrue((ROOT / "deployment/requirements.txt").is_file())

    def test_active_imports(self):
        migrated = set()
        for path in active_python():
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                names = ([node.module or ""] if isinstance(node, ast.ImportFrom)
                         else [alias.name for alias in node.names] if isinstance(node, ast.Import)
                         else [])
                for name in names:
                    self.assertNotIn(name.split(".")[0], ("final_training", "dashboard", "scripts", "submission"), str(path))
                    if name.startswith("final_model."):
                        migrated.add(name)
                        module = ROOT.joinpath(*name.split("."))
                        self.assertTrue(module.with_suffix(".py").is_file() or (module / "__init__.py").is_file())
        self.assertTrue({"final_model.inference", "final_model.training_core",
                         "final_model.demo_models", "final_model.demo_registry"} <= migrated)

    def test_slurm_python_targets(self):
        targets = []
        for path in (ROOT / "experiments").rglob("*"):
            if path.suffix not in (".slurm", ".sh"):
                continue
            for match in re.finditer(r"^\s*python(?:3(?:\.\d+)?)?\s+([^\s]+\.py)\b", path.read_text(), re.M):
                target = ROOT / match.group(1)
                targets.append(target)
                self.assertTrue(target.is_file(), f"{path}: {target}")
        self.assertTrue(targets)

    def test_registry_paths(self):
        path = ROOT / "final_model/demo_registry.py"
        spec = importlib.util.spec_from_file_location("structure_registry", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.PROJECT_ROOT, ROOT)
        self.assertTrue(module.validate_demo_models())
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

        visit(json.loads((ROOT / "web/data/catalogue.json").read_text()))
        self.assertTrue(references)

    def test_docker_copy_sources(self):
        dockerfile = (ROOT / "deployment/Dockerfile").read_text()
        self.assertIn("COPY final_model /app/final_model", dockerfile)
        self.assertNotIn("requirements-api.txt", dockerfile)
        for line in dockerfile.splitlines():
            if line.startswith("COPY "):
                for source in shlex.split(line)[1:-1]:
                    self.assertTrue((ROOT / source).exists(), source)


if __name__ == "__main__":
    unittest.main()
