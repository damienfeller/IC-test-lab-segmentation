"""Configuration helpers for nnU-Net style training runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional
import json

try:  # pragma: no cover - trivial import guard
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


def _load_mapping(path: Path) -> Mapping[str, Any]:
    """Load a mapping from a YAML or JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file '{path}' does not exist.")

    suffix = path.suffix.lower()
    with path.open("r", encoding="utf8") as fh:
        if suffix in {".yml", ".yaml"}:
            if yaml is None:
                raise ModuleNotFoundError(
                    "PyYAML is required to load YAML configuration files. "
                    "Install it or use JSON instead."
                )
            data = yaml.safe_load(fh)
        elif suffix == ".json":
            data = json.load(fh)
        else:
            raise ValueError(
                "Unsupported configuration format. Expected a YAML or JSON file, "
                f"got '{path.suffix}'."
            )

    if not isinstance(data, MutableMapping):
        raise TypeError(
            "Configuration files must define a mapping of settings. "
            f"Found {type(data).__name__}."
        )
    return data


@dataclass(frozen=True)
class TrainerConfig:
    """Dataclass describing an nnU-Net training configuration.

    The class provides convenience constructors for loading configuration values
    from YAML or JSON files. The configuration is intentionally minimal yet
    expressive enough to re-create nnU-Net style experiments. Additional options
    can be stored within the ``extra`` dictionary.
    """

    dataset_root: Path
    output_root: Path
    plan_name: str
    configuration: str
    trainer: str = "nnUNetTrainer"
    folds: Iterable[int] = field(default_factory=lambda: (0,))
    gpu_ids: Optional[Iterable[int]] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TrainerConfig":
        """Create a configuration instance from a mapping."""
        required_keys = {"dataset_root", "output_root", "plan_name", "configuration"}
        missing = required_keys - mapping.keys()
        if missing:
            missing_keys = ", ".join(sorted(missing))
            raise KeyError(f"Missing required configuration keys: {missing_keys}")

        dataset_root = Path(mapping["dataset_root"]).expanduser().resolve()
        output_root = Path(mapping["output_root"]).expanduser().resolve()
        plan_name = str(mapping["plan_name"])
        configuration = str(mapping["configuration"])
        trainer = str(mapping.get("trainer", "nnUNetTrainer"))

        folds = mapping.get("folds", (0,))
        if isinstance(folds, int):
            folds = (folds,)
        elif isinstance(folds, Iterable) and not isinstance(folds, (str, bytes)):
            folds = tuple(int(f) for f in folds)
        else:
            raise TypeError("'folds' must be an integer or iterable of integers.")

        gpu_ids = mapping.get("gpu_ids")
        if gpu_ids is not None:
            if isinstance(gpu_ids, int):
                gpu_ids = (gpu_ids,)
            elif isinstance(gpu_ids, Iterable) and not isinstance(gpu_ids, (str, bytes)):
                gpu_ids = tuple(int(g) for g in gpu_ids)
            else:
                raise TypeError("'gpu_ids' must be an integer or iterable of integers.")

        extra = {
            key: value
            for key, value in mapping.items()
            if key not in {
                "dataset_root",
                "output_root",
                "plan_name",
                "configuration",
                "trainer",
                "folds",
                "gpu_ids",
            }
        }
        return cls(
            dataset_root=dataset_root,
            output_root=output_root,
            plan_name=plan_name,
            configuration=configuration,
            trainer=trainer,
            folds=folds,
            gpu_ids=gpu_ids,
            extra=extra,
        )

    @classmethod
    def from_file(cls, path: Path | str) -> "TrainerConfig":
        """Create a configuration instance from a YAML or JSON file."""
        config_path = Path(path)
        mapping = _load_mapping(config_path)
        return cls.from_mapping(mapping)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration into a serialisable dictionary."""
        data: Dict[str, Any] = {
            "dataset_root": str(self.dataset_root),
            "output_root": str(self.output_root),
            "plan_name": self.plan_name,
            "configuration": self.configuration,
            "trainer": self.trainer,
            "folds": list(self.folds),
        }
        if self.gpu_ids is not None:
            data["gpu_ids"] = list(self.gpu_ids)
        if self.extra:
            data.update(self.extra)
        return data

    def dump(self, path: Path | str) -> None:
        """Write the configuration to the given path in YAML format."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf8") as fh:
            if yaml is None:
                json.dump(self.to_dict(), fh, indent=2)
            else:
                yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    def describe(self) -> str:
        """Return a human readable summary of the configuration."""
        gpu_info = (
            "auto" if self.gpu_ids is None else ", ".join(str(gpu) for gpu in self.gpu_ids)
        )
        return (
            f"Trainer: {self.trainer}\n"
            f"Plan: {self.plan_name}\n"
            f"Configuration: {self.configuration}\n"
            f"Dataset: {self.dataset_root}\n"
            f"Output: {self.output_root}\n"
            f"Folds: {', '.join(str(fold) for fold in self.folds)}\n"
            f"GPUs: {gpu_info}"
        )
