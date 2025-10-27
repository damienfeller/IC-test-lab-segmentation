"""Helpers for creating and validating nnU-Net experiment plans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
import json

try:  # pragma: no cover - see config module for rationale
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


@dataclass
class ExperimentPlan:
    """A lightweight representation of an experiment plan."""

    name: str
    modalities: List[str]
    target_labels: List[str]
    spacing: Iterable[float]
    patch_size: Iterable[int]
    batch_size: int
    augmentations: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "modalities": list(self.modalities),
            "target_labels": list(self.target_labels),
            "spacing": list(self.spacing),
            "patch_size": list(self.patch_size),
            "batch_size": int(self.batch_size),
            "augmentations": dict(self.augmentations or {}),
        }


class ExperimentPlanBuilder:
    """Utility class to build and persist experiment plans.

    The nnU-Net framework stores experiment plans as YAML files that describe how
    data should be preprocessed and batched. The builder provides a structured
    interface to create these plans in Python code.
    """

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_plan(self, plan: ExperimentPlan, *, file_name: Optional[str] = None) -> Path:
        """Write the given plan to disk and return the resulting path."""
        plan_dict = plan.to_dict()
        path = self.output_dir / f"{file_name or plan.name}.yaml"
        with path.open("w", encoding="utf8") as fh:
            if yaml is None:
                json.dump(plan_dict, fh, indent=2)
            else:
                yaml.safe_dump(plan_dict, fh, sort_keys=False)
        return path

    def load_plan(self, path: Path | str) -> ExperimentPlan:
        """Load an experiment plan from YAML or JSON."""
        plan_path = Path(path)
        if not plan_path.exists():
            raise FileNotFoundError(f"Experiment plan '{plan_path}' does not exist.")

        suffix = plan_path.suffix.lower()
        with plan_path.open("r", encoding="utf8") as fh:
            if suffix in {".yml", ".yaml"}:
                if yaml is None:
                    raise ModuleNotFoundError(
                        "PyYAML is required to load YAML experiment plans."
                    )
                data = yaml.safe_load(fh)
            elif suffix == ".json":
                data = json.load(fh)
            else:
                raise ValueError(
                    f"Unsupported experiment plan format: '{plan_path.suffix}'."
                )

        return ExperimentPlan(
            name=data["name"],
            modalities=list(data["modalities"]),
            target_labels=list(data["target_labels"]),
            spacing=data["spacing"],
            patch_size=data["patch_size"],
            batch_size=int(data["batch_size"]),
            augmentations=data.get("augmentations"),
        )

    def validate(self, plan: ExperimentPlan) -> None:
        """Perform a basic validation on the provided plan."""
        if plan.batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        if len(list(plan.spacing)) != 3:
            raise ValueError("Spacing must contain three values (x, y, z).")
        if len(list(plan.patch_size)) != 3:
            raise ValueError("Patch size must contain three values (x, y, z).")
        if not plan.modalities:
            raise ValueError("At least one modality is required.")
        if not plan.target_labels:
            raise ValueError("At least one target label is required.")


__all__ = ["ExperimentPlan", "ExperimentPlanBuilder"]
