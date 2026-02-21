from __future__ import annotations

from .base import ErrorModel
from .bootstrap import BootstrapErrorModel
from .dimsum_analog import DiMSumAnalogErrorModel, DropletFullErrorModel


def get_error_model(name: str) -> ErrorModel:
    normalized = (name or "bootstrap").strip().lower()
    if normalized == "bootstrap":
        return BootstrapErrorModel()
    if normalized in {"dimsum_analog", "dimsum", "analog"}:
        return DiMSumAnalogErrorModel()
    if normalized in {"droplet_full", "droplet", "full"}:
        return DropletFullErrorModel()
    raise ValueError(f"Unknown error model: {name}")
