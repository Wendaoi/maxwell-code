"""Session completeness checks for Pong recording directories."""

from __future__ import annotations

from pathlib import Path


REQUIRED_FILES = (
    "session_manifest.json",
    "session_config.json",
    "resolved_layout.json",
    "runtime_events.jsonl",
    "window_samples.csv",
    "quality_summary.json",
)


def validate_session_dir(session_dir: str | Path) -> dict[str, object]:
    """Check that raw data, layout, runtime logs, and metadata are present."""
    session_path = Path(session_dir)
    missing = [
        filename
        for filename in REQUIRED_FILES
        if not (session_path / filename).exists()
    ]
    raw_files = sorted(path.name for path in session_path.glob("*.raw.h5"))
    if not raw_files:
        missing.append("*.raw.h5")

    return {
        "session_dir": str(session_path),
        "valid": not missing,
        "missing": missing,
        "raw_files": raw_files,
    }
