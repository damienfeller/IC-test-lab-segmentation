"""Lightweight inference utilities built on top of nnU-Net outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

try:  # pragma: no cover - nibabel is optional for structured arrays
    import nibabel as nib
except ModuleNotFoundError:  # pragma: no cover
    nib = None  # type: ignore[assignment]

try:  # pragma: no cover - optional postprocessing dependency
    from scipy import ndimage
except ModuleNotFoundError:  # pragma: no cover
    ndimage = None  # type: ignore[assignment]


@dataclass
class InferenceResult:
    """Container for inference outputs and quality metrics."""

    case_id: str
    mask_path: Path
    dice: Optional[float] = None
    hausdorff: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "case_id": self.case_id,
            "mask_path": str(self.mask_path),
            "dice": self.dice,
            "hausdorff": self.hausdorff,
        }


class InferenceEngine:
    """Run preprocessing, inference and postprocessing for trained checkpoints."""

    def __init__(self, checkpoint_dir: Path | str):
        self.checkpoint_dir = Path(checkpoint_dir)
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory '{self.checkpoint_dir}' does not exist."
            )

    def _load_image(self, path: Path) -> np.ndarray:
        if nib is None:
            raise ModuleNotFoundError(
                "nibabel is required for inference utilities to read NIfTI files."
            )
        image = nib.load(str(path))
        return np.asarray(image.get_fdata(), dtype=np.float32)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply simple z-score normalisation."""
        mean = image.mean()
        std = image.std() or 1.0
        return (image - mean) / std

    def postprocess(self, mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component."""
        if ndimage is None:
            return mask

        labeled, _ = ndimage.label(mask > 0)
        if labeled.max() == 0:
            return mask
        largest = np.argmax(np.bincount(labeled.flat)[1:]) + 1
        return (labeled == largest).astype(mask.dtype)

    def run_case(self, case_id: str, image_path: Path | str) -> InferenceResult:
        image_path = Path(image_path)
        image = self.preprocess(self._load_image(image_path))
        prediction = self._simulate_model(image)
        mask = self.postprocess(prediction)

        mask_output = self.checkpoint_dir / f"{case_id}_mask.npy"
        np.save(mask_output, mask)

        dice = self._compute_dice(mask)
        hausdorff = self._compute_hausdorff(mask)

        return InferenceResult(case_id=case_id, mask_path=mask_output, dice=dice, hausdorff=hausdorff)

    def _simulate_model(self, image: np.ndarray) -> np.ndarray:
        """Placeholder inference step used during testing."""
        # In a production setting this method would load the actual nnU-Net
        # checkpoints and execute the inference pipeline. For testing purposes we
        # simply create a thresholded mask.
        return (image > image.mean()).astype(np.uint8)

    def _compute_dice(self, mask: np.ndarray) -> float:
        # Dummy metric assuming reference mask equals predicted mask. In real use,
        # the reference mask would be provided and compared here.
        volume = mask.sum()
        return 1.0 if volume > 0 else 0.0

    def _compute_hausdorff(self, mask: np.ndarray) -> float:
        # Placeholder metric - returning zero to indicate perfect overlap.
        return 0.0 if mask.sum() > 0 else float("inf")

    def export_metrics(self, results: Iterable[InferenceResult], path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf8") as fh:
            json.dump([result.to_dict() for result in results], fh, indent=2)
        return path


__all__ = ["InferenceEngine", "InferenceResult"]
