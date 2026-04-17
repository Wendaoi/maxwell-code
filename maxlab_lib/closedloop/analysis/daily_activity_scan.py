"""Manifest helpers for daily checkerboard activity scans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def create_daily_scan_manifest(
    root: str,
    culture_id: str,
    div: int,
    scan_paths: Iterable[str],
    timestamp: str,
) -> str:
    """Record the raw files and fixed parameters for a daily checkerboard scan."""
    scan_paths = list(scan_paths)
    if len(scan_paths) != 14:
        raise ValueError("daily checkerboard scan requires 14 configuration files")

    output_dir = Path(root) / f"{culture_id}_div{div:03d}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "daily_activity_scan_manifest.json"
    manifest = {
        "culture_id": culture_id,
        "div": div,
        "timestamp": timestamp,
        "checkerboard_configurations": 14,
        "record_seconds_per_configuration": 15,
        "highpass_hz": 300,
        "gain": 512,
        "threshold_sigma": 6,
        "scan_paths": scan_paths,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return str(manifest_path)
