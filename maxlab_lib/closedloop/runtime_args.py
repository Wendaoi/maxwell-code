"""Runtime argument helpers for the closed-loop Pong experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

_CONDITION_ALIASES = {
    "STIM": "STIM",
    "STIMULUS": "STIM",
    "SILENT": "SILENT",
    "NO_FEEDBACK": "NO_FEEDBACK",
    "NO-FEEDBACK": "NO_FEEDBACK",
    "NOFEEDBACK": "NO_FEEDBACK",
    "REST": "REST",
}


def normalize_condition(condition: str) -> str:
    """Normalize experiment condition labels used by the Python runtime."""
    normalized = condition.strip().upper()
    if normalized == "RANDOM_STIMULUS":
        raise ValueError("RANDOM_STIMULUS is not a supported runtime condition")
    try:
        return _CONDITION_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported condition: {condition}") from exc


def generate_cpp_args(wells: Iterable[int], config_path: str | Path) -> List[str]:
    """Generate argv values for the C++ closed-loop runtime."""
    wells_list = list(wells)
    if len(wells_list) > 1:
        raise ValueError("Closed-loop Pong runtime supports exactly one target well")
    target_well = wells_list[0] if wells_list else 0
    return [
        str(config_path),
        str(target_well),
        "1",
        "1",
    ]
