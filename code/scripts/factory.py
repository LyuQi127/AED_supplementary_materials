from __future__ import annotations

import argparse


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-arguments", nargs="*", default=[])


def add_device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cpu")


def build_model(*args, **kwargs):
    raise NotImplementedError


def resolve_dtype(name: str):
    return name
