"""Command line entry points for nnU-Net training and evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import TrainerConfig
from .trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Launch nnU-Net training")
    train_parser.add_argument(
        "config",
        type=Path,
        help="Path to a YAML or JSON configuration file compatible with TrainerConfig.",
    )

    eval_parser = subparsers.add_parser("eval", help="Run inference/evaluation")
    eval_parser.add_argument(
        "config",
        type=Path,
        help="Path to a YAML or JSON configuration file compatible with TrainerConfig.",
    )
    eval_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to evaluate. If omitted, nnU-Net will use its defaults.",
    )

    return parser


def _execute_train(args: argparse.Namespace) -> None:
    config = TrainerConfig.from_file(args.config)
    trainer = Trainer(config)
    trainer.train()


def _execute_eval(args: argparse.Namespace) -> None:
    config = TrainerConfig.from_file(args.config)
    trainer = Trainer(config)
    trainer.evaluate(checkpoint=args.checkpoint)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        _execute_train(args)
    elif args.command == "eval":
        _execute_eval(args)
    else:  # pragma: no cover - argparse enforces the command choices
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
