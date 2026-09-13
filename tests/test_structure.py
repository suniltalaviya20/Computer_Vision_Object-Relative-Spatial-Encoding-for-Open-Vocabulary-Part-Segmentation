"""Repository layout checks: standard library only, no encoders or GPU."""

import ast
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DIRS = ("final_model", "tools", "datasets", "experiments", "deployment")
IMPLEMENTATION_FAMILIES = ("features", "baseline_segmentation", "geometry_comparison",
                           "part_query_alignment", "object_centric_zoom", "query_gated_uvd", "robustness")
RUNNER_FAMILIES = ("analysis", "feature_extraction", "baseline_segmentation", "geometry_comparison",
                   "part_query_alignment", "object_centric_zoom", "query_gated_uvd", "robustness", "crop_alignment_uvd")


def active_python():
    yield ROOT / "inference_server.py"
    yield ROOT / "inference_options.py"
    for directory in ACTIVE_DIRS:
        yield from (ROOT / directory).rglob("*.py")


class StructureTests(unittest.TestCase):
    def test_layout(self):
        for directory in (*ACTIVE_DIRS, "web", "tests", "final_training_notebooks",
                          "scripts", "data", "models", "training_results",
                          "experiments/implementations", "experiments/runners"):
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
        for family in IMPLEMENTATION_FAMILIES:
            self.assertTrue((ROOT / "experiments/implementations" / family).is_dir())
        for family in RUNNER_FAMILIES:
            self.assertTrue((ROOT / "experiments/runners" / family).is_dir())
            self.assertFalse((ROOT / "experiments" / family).exists())
        for old in ("src", "dashboard", "deployment/requirements-api.txt"):
            self.assertFalse((ROOT / old).exists(), old)
        self.assertFalse(any((ROOT / "final_training").glob("*.py")))
        self.assertTrue((ROOT / "deployment/requirements.txt").is_file())

    def test_active_imports(self):
        migrated = set()
        for path in active_python():
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    for family in RUNNER_FAMILIES:
                        self.assertNotIn(f"python experiments/{family}/", node.value, str(path))
                names = ([node.module or ""] if isinstance(node, ast.ImportFrom)
                         else [alias.name for alias in node.names] if isinstance(node, ast.Import)
                         else [])
                for name in names:
                    self.assertNotIn(name.split(".")[0], ("src", "final_training", "dashboard", "scripts", "submission"), str(path))
                    if name.startswith("experiments."):
                        self.assertIn(name.split(".")[1], ("implementations", "runners"), str(path))
                        module = ROOT.joinpath(*name.split("."))
                        self.assertTrue(module.with_suffix(".py").is_file() or module.is_dir(), name)
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

    def test_runner_roots(self):
        runners = list((ROOT / "experiments/runners").rglob("*.py"))
        self.assertTrue(runners)
        for path in runners:
            tree = ast.parse(path.read_text())
            assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == "PROJECT_ROOT"
                                   for target in node.targets)]
            self.assertEqual(len(assignments), 1, str(path))
            expression = ast.Expression(assignments[0].value)
            resolved = eval(compile(expression, str(path), "eval"),
                            {"Path": Path, "__file__": str(path)})
            self.assertEqual(resolved, ROOT, str(path))

    def test_training_artifacts_unchanged(self):
        changed = subprocess.check_output(
            ["git", "diff", "HEAD", "--name-only", "--", "final_training_notebooks/", "training_results/"], cwd=ROOT, text=True
        )
        self.assertEqual(changed, "", changed)

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
