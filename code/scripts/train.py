from __future__ import annotations

import argparse
from pathlib import Path

from .factory import add_device_argument, add_model_arguments


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--data-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--batch-size", type=int, required=True)
    result.add_argument("--steps", type=int, required=True)
    result.add_argument("--learning-rate", type=float, required=True)
    result.add_argument("--transition-weight", type=float, default=0.01)
    add_device_argument(result)
    add_model_arguments(result)
    return result


def main() -> None:
    parser().parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
