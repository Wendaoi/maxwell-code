# Closed-loop Pong Analysis

These helpers turn one completed session directory into behavior and electrophysiology summaries that match the data products described in `Pong game 实验复现日程表.md`.

All commands below assume you are running from the `PongforMaxone/` workspace root:

```bash
cd maxwell-code/maxlab_lib/closedloop
```

## What Lives In This Directory

| Module | Purpose |
| --- | --- |
| `behavior_metrics.py` | Computes replayable behavior metrics from `runtime_events.jsonl` and `window_samples.csv` |
| `daily_activity_scan.py` | Writes a manifest for the daily 14-configuration checkerboard scan used for Figure 3 |
| `ephys_metrics.py` | Lightweight helpers for cross-correlation, entropy, center-of-activity, and functional plasticity |
| `extract_spikes.py` | Negative threshold-crossing helper for post-hoc raw trace processing |
| `validate_session.py` | Checks whether a session directory contains all required runtime outputs |

## Expected Session Inputs

Replace `SESSION_DIR` below with one experiment session folder. A valid session should contain:

- `session_manifest.json`
- `session_config.json`
- `resolved_layout.json`
- `runtime_events.jsonl`
- `window_samples.csv`
- `quality_summary.json`
- `*.raw.h5`

## 1. Validate Session Completeness

Run this first after a recording finishes:

```bash
SESSION_DIR="$HOME/pong_experiments/<session_id>"

python3 - <<'PY'
import json
import os
from analysis.validate_session import validate_session_dir

session_dir = os.environ["SESSION_DIR"]
print(json.dumps(validate_session_dir(session_dir), indent=2))
PY
```

`"valid": true` means the expected metadata, logs, and raw H5 file are all present.

## 2. Compute Behavior Metrics

This covers the behavior outputs needed for Figure 5, Figure 6, and Figure S4:

- Average Rally Length
- Aces
- Long Rallies
- Paddle Movement

The helper automatically supports the paper-aligned exclusion of the first `10 s` of gameplay:

```bash
SESSION_DIR="$HOME/pong_experiments/<session_id>"

python3 - <<'PY'
import csv
import json
import os
from pathlib import Path
from analysis.behavior_metrics import compute_behavior_metrics

session_dir = Path(os.environ["SESSION_DIR"])

with open(session_dir / "runtime_events.jsonl") as f:
    events = [json.loads(line) for line in f if line.strip()]

with open(session_dir / "window_samples.csv", newline="") as f:
    windows = list(csv.DictReader(f))

metrics = compute_behavior_metrics(
    windows,
    events,
    exclude_initial_game_seconds=10,
)
print(json.dumps(metrics, indent=2))
PY
```

## 3. Register Daily Checkerboard Scans

Use this for the maturation-stage daily scan that feeds Figure 3 bookkeeping:

```bash
python3 - <<'PY'
from analysis.daily_activity_scan import create_daily_scan_manifest

manifest_path = create_daily_scan_manifest(
    root="$HOME/pong_experiments/daily_scans",
    culture_id="culture_a",
    div=14,
    scan_paths=[
        "/path/to/checkerboard_config_01.raw.h5",
        "/path/to/checkerboard_config_02.raw.h5",
        "/path/to/checkerboard_config_03.raw.h5",
        "/path/to/checkerboard_config_04.raw.h5",
        "/path/to/checkerboard_config_05.raw.h5",
        "/path/to/checkerboard_config_06.raw.h5",
        "/path/to/checkerboard_config_07.raw.h5",
        "/path/to/checkerboard_config_08.raw.h5",
        "/path/to/checkerboard_config_09.raw.h5",
        "/path/to/checkerboard_config_10.raw.h5",
        "/path/to/checkerboard_config_11.raw.h5",
        "/path/to/checkerboard_config_12.raw.h5",
        "/path/to/checkerboard_config_13.raw.h5",
        "/path/to/checkerboard_config_14.raw.h5",
    ],
    timestamp="20260416",
)
print(manifest_path)
PY
```

The manifest hardcodes the schedule-required acquisition assumptions:

| Field | Value |
| --- | --- |
| Configurations | `14` |
| Record time / config | `15 s` |
| High-pass | `300 Hz` |
| Gain | `512x` |
| Threshold | `6 sigma` |

## 4. Use Ephys Helper Functions

These helpers are intended for post-hoc pipelines after you have binned spikes from `.raw.h5` or a spike table:

```bash
python3 - <<'PY'
from analysis.ephys_metrics import (
    binary_entropy,
    center_of_activity,
    cluster_binary_entropy,
    cross_correlation,
    exclusive_motor_event_percentage,
    functional_plasticity_distance,
)

print("cross_correlation:", cross_correlation([1, 2, 3], [1, 2, 3]))
print("cluster_entropy:", cluster_binary_entropy([0, 1, 1, 0]))
print("binary_entropy:", binary_entropy(0.5))

ca = center_of_activity([
    {"spikes": 1, "x": 0, "y": 0},
    {"spikes": 3, "x": 4, "y": 8},
])
print("center_of_activity:", ca)
print("plasticity_distance:", functional_plasticity_distance(ca, (0, 0)))

exclusive = exclusive_motor_event_percentage([
    {"motor_1_spikes": 2, "motor_2_spikes": 0},
    {"motor_1_spikes": 0, "motor_2_spikes": 3},
    {"motor_1_spikes": 1, "motor_2_spikes": 1},
    {"motor_1_spikes": 0, "motor_2_spikes": 0},
])
print("exclusive_motor_event_percentage:", exclusive)
PY
```

Recommended binning conventions from the schedule:

| Metric | Recommended input bin |
| --- | --- |
| Cross-correlation | `100 ms` |
| Information entropy | `100 ms` |
| Exclusive motor region activity | `1000 ms` |
| Functional plasticity / CA trajectory | `5 min` windows compared to the matched `10 min` pre-rest centroid |

## 5. Detect Spikes From Raw Traces

`detect_threshold_crossings()` is a simple post-hoc helper for channel traces already loaded into Python:

```bash
python3 - <<'PY'
from analysis.extract_spikes import detect_threshold_crossings

traces = {
    3: [0.0, -1.0, -6.0, -2.0, -7.0],
    4: [0.0, 0.0, -2.0, -3.0, -4.0],
}

spikes = detect_threshold_crossings(
    traces,
    threshold=-5.0,
    refractory_samples=2,
)
print(spikes)
PY
```

## 6. Run Analysis-Related Tests

Run these after editing anything in `analysis/`, `layout_config.py`, or the logging schema:

```bash
python3 -m unittest \
  maxwell-code/maxlab_lib/closedloop/tests/test_analysis_tools.py \
  maxwell-code/maxlab_lib/closedloop/tests/test_layout_config.py \
  maxwell-code/maxlab_lib/closedloop/tests/test_pong_setup_analysis.py
```

For the C++ runtime helpers:

```bash
cd maxwell-code/maxlab_lib
make test_runtime_config test_runtime_timing test_runtime_logging test_motor_decoder test_ponggame USE_QT=0
```
