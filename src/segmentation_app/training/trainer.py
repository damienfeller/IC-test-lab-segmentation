"""High level training orchestrator wrapping nnU-Net utilities."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, Optional

from .config import TrainerConfig


class Trainer:
    """Orchestrates nnU-Net training and evaluation pipelines.

    The class intentionally avoids depending on nnU-Net internals directly. It
    instead shells out to the ``nnUNetv2_train`` and ``nnUNetv2_predict`` command
    line interfaces when available. This makes the implementation lightweight and
    keeps the integration points clearly defined.
    """

    def __init__(self, config: TrainerConfig):
        self.config = config

    def _run_command(self, command: Iterable[str]) -> None:
        """Execute a command, raising an informative error on failure."""
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as exc:  # pragma: no cover - requires external binary
            raise RuntimeError(
                "Required nnU-Net command not found. Ensure nnU-Net is installed and "
                "available on PATH."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Command '{' '.join(command)}' failed with exit code {exc.returncode}."
            ) from exc

    def train(self) -> None:
        """Launch nnU-Net training for the configured folds."""
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        for fold in self.config.folds:
            command = [
                "nnUNetv2_train",
                self.config.plan_name,
                self.config.configuration,
                str(fold),
                "-tr",
                self.config.trainer,
                "-p",
                self.config.plan_name,
            ]
            if self.config.gpu_ids is not None:
                command += ["-g", ",".join(str(g) for g in self.config.gpu_ids)]
            if "num_epochs" in self.config.extra:
                command += ["--num_epochs", str(self.config.extra["num_epochs"])]
            self._run_command(command)

    def evaluate(self, checkpoint: Optional[Path | str] = None) -> None:
        """Run evaluation using ``nnUNetv2_predict``."""
        inference_dir = self.config.output_root / "inference"
        inference_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "nnUNetv2_predict",
            "-d",
            str(self.config.dataset_root),
            "-o",
            str(inference_dir),
            "-p",
            self.config.plan_name,
            "-c",
            self.config.configuration,
        ]
        if checkpoint is not None:
            command += ["-chk", str(checkpoint)]
        if self.config.gpu_ids is not None:
            command += ["-g", ",".join(str(g) for g in self.config.gpu_ids)]
        self._run_command(command)


__all__ = ["Trainer"]
