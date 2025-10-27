"""Utilities for creating and managing dataset splits."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence

try:  # pragma: no cover
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


@dataclass
class DataSplit:
    """Container describing a set of dataset splits."""

    train: List[str]
    validation: List[str]
    test: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, List[str]]:
        data = {"train": list(self.train), "validation": list(self.validation)}
        if self.test is not None:
            data["test"] = list(self.test)
        return data


class DataSplitManager:
    """Manage storage and generation of dataset splits."""

    def __init__(self, dataset_root: Path | str):
        self.dataset_root = Path(dataset_root)
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root '{self.dataset_root}' does not exist.")

    def list_cases(self, pattern: str = "*.nii.gz") -> List[str]:
        """Return available cases relative to the dataset root."""
        return sorted(str(path.relative_to(self.dataset_root)) for path in self.dataset_root.glob(pattern))

    def create_random_split(
        self,
        *,
        cases: Optional[Sequence[str]] = None,
        validation_ratio: float = 0.2,
        test_ratio: float = 0.0,
        seed: Optional[int] = None,
    ) -> DataSplit:
        """Create a random split of the available cases."""
        if cases is None:
            cases = self.list_cases()
        cases = list(cases)
        if not cases:
            raise ValueError("No cases were provided to create a split.")
        if seed is not None:
            random.seed(seed)
        random.shuffle(cases)

        total = len(cases)
        val_count = int(total * validation_ratio)
        test_count = int(total * test_ratio)

        validation = cases[:val_count]
        test = cases[val_count : val_count + test_count] if test_count > 0 else []
        train = cases[val_count + test_count :]

        if not train or not validation:
            raise ValueError("Both train and validation splits must contain cases.")

        return DataSplit(train=train, validation=validation, test=test or None)

    def save(self, split: DataSplit, path: Path | str) -> Path:
        """Persist the split as YAML (if available) or JSON."""
        split_path = Path(path)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with split_path.open("w", encoding="utf8") as fh:
            if split_path.suffix.lower() in {".yml", ".yaml"} and yaml is not None:
                yaml.safe_dump(split.to_dict(), fh, sort_keys=False)
            else:
                json.dump(split.to_dict(), fh, indent=2)
        return split_path

    def load(self, path: Path | str) -> DataSplit:
        split_path = Path(path)
        if not split_path.exists():
            raise FileNotFoundError(f"Split file '{split_path}' does not exist.")

        with split_path.open("r", encoding="utf8") as fh:
            if split_path.suffix.lower() in {".yml", ".yaml"}:
                if yaml is None:
                    raise ModuleNotFoundError("PyYAML is required to load YAML split files.")
                data = yaml.safe_load(fh)
            else:
                data = json.load(fh)

        if not isinstance(data, MutableMapping):
            raise TypeError("Split files must contain a mapping of split names.")
        if "train" not in data or "validation" not in data:
            raise KeyError("Split must define 'train' and 'validation' keys.")

        return DataSplit(
            train=list(data["train"]),
            validation=list(data["validation"]),
            test=list(data["test"]) if data.get("test") else None,
        )


__all__ = ["DataSplit", "DataSplitManager"]
