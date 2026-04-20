# tests

This directory contains the regression suite for the closed-loop Pong runtime. The tests are split between focused C++ unit-style binaries and Python tests for the orchestration and analysis helpers.

## Test Inventory

| Test file | Language | What it protects |
| --- | --- | --- |
| `ponggame_test.cpp` | C++ | Game geometry, serve-speed assumptions, sensory-zone mapping, and rally-bounce semantics |
| `motor_decoder_test.cpp` | C++ | Baseline gain computation, clamping, and identity fallbacks |
| `spike_detection_test.cpp` | C++ | Paper-aligned raw detector behavior: high-pass activity, MAD thresholding, refractory handling, and median side-path tracking |
| `runtime_config_test.cpp` | C++ | JSON config parsing, condition aliases, runtime schema validation, and sequence lookup |
| `runtime_logging_test.cpp` | C++ | Event log, window CSV, and quality summary output format |
| `runtime_timing_test.cpp` | C++ | Sample-count math and phase elapsed-time helpers |
| `gamewindow_test.cpp` | C++ / Qt | GUI rendering alignment with `PongGame` geometry |
| `test_analysis_tools.py` | Python | Behavior metrics, daily scan manifests, ephys helpers, spike extraction, and session validation |
| `test_layout_config.py` | Python | Electrode layout, condition normalization, and `pong_setup.py` helper behavior with fake MaxLab bindings |
| `test_pong_setup_analysis.py` | Python | Post-run analysis fallback behavior when H5 event tables are empty |

## Recommended Commands

Run the non-GUI C++ tests from `maxwell-code/maxlab_lib`:

```bash
cd maxwell-code/maxlab_lib
make test_spike_detection test_runtime_config test_runtime_timing test_runtime_logging test_motor_decoder test_ponggame USE_QT=0
```

Run the Qt rendering test only on machines with Qt6 available:

```bash
cd maxwell-code/maxlab_lib
make test_gamewindow USE_QT=1
```

Run the Python tests from the repository root:

```bash
conda run -n base python -m unittest \
  maxwell-code/maxlab_lib/closedloop/tests/test_analysis_tools.py \
  maxwell-code/maxlab_lib/closedloop/tests/test_layout_config.py \
  maxwell-code/maxlab_lib/closedloop/tests/test_pong_setup_analysis.py
```

## When To Run What

| If you changed... | Run at least... |
| --- | --- |
| `ponggame.cpp` / `ponggame.h` | `make test_ponggame USE_QT=0` |
| `motor_decoder.cpp` / `motor_decoder.h` | `make test_motor_decoder USE_QT=0` |
| `spike_detection.cpp` / `spike_detection.h` / `filter.cpp` / `filter.h` | `make test_spike_detection USE_QT=0` |
| `runtime_config.cpp` / `runtime_config.h` / `pong_setup.py` config schema | `make test_runtime_config USE_QT=0` and `conda run -n base python -m unittest .../test_layout_config.py` |
| `runtime_logging.cpp` / analysis log consumers | `make test_runtime_logging USE_QT=0` and `conda run -n base python -m unittest .../test_analysis_tools.py` |
| `runtime_timing.cpp` | `make test_runtime_timing USE_QT=0` |
| `gamewindow.cpp` | `make test_gamewindow USE_QT=1` |
| `analysis/*.py` | `conda run -n base python -m unittest .../test_analysis_tools.py` |

## Notes

- The C++ tests are lightweight standalone binaries; they do not require connected hardware.
- `test_layout_config.py` uses fake `maxlab`, `numpy`, and `h5py` shims so it can exercise setup helpers without a live MaxLab installation.
- GUI verification uses `QT_QPA_PLATFORM=offscreen` in the Makefile target, so it can run in headless environments as long as Qt6 is installed.
