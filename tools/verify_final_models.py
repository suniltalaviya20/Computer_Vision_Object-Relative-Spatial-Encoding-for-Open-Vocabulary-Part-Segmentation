#!/usr/bin/env python3
"""Validate final-study local-UI checkpoints without loading the encoders."""

import hashlib
import json
from pathlib import Path

import torch
from torch.torch_version import TorchVersion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "models" / "final_study"
REGISTRY_PATH = MODEL_ROOT / "model_registry.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    registry = json.loads(REGISTRY_PATH.read_text())
    errors = []

    with torch.serialization.safe_globals([TorchVersion]):
        for experiment, entry in registry["models"].items():
            path = PROJECT_ROOT / entry["checkpoint"]
            if not path.is_file():
                errors.append(f"{experiment}: missing {path}")
                continue
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint.get("format_version") != 1:
                errors.append(f"{experiment}: unsupported format")
            if checkpoint.get("experiment") != experiment:
                errors.append(
                    f"{experiment}: checkpoint contains {checkpoint.get('experiment')!r}"
                )
            if "model_state" not in checkpoint:
                errors.append(f"{experiment}: model_state is missing")
            print(
                f"OK  {experiment:<24} epoch={checkpoint.get('selected_epoch')} "
                f"validation_iou={entry['validation_iou']:.4f}"
            )

    selected = PROJECT_ROOT / registry["selected_checkpoint"]
    selected_source = PROJECT_ROOT / registry["models"][registry["selected_model"]]["checkpoint"]
    if not selected.is_file() or sha256(selected) != sha256(selected_source):
        errors.append("best_model.pt does not match the selected model")

    if errors:
        raise SystemExit("\n".join(f"ERROR  {error}" for error in errors))
    print(f"\nSelected model: {registry['selected_model']}")
    print("All local-UI checkpoints are valid.")


if __name__ == "__main__":
    main()
