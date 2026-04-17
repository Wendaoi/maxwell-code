# include

This directory holds the C++ headers that define the closed-loop runtime contract. If you need to understand which module owns a piece of logic before diving into the `.cpp` files, start here.

## Header Map

| Header | Primary implementation | Responsibility |
| --- | --- | --- |
| `ponggame.h` | `../ponggame.cpp` | Pong state machine, game geometry, hit/miss events, and condition-dependent sensory-zone logic |
| `motor_decoder.h` | `../motor_decoder.cpp` | Converts grouped spike counts into normalized up/down motor activity using baseline-derived gains |
| `runtime_config.h` | `../runtime_config.cpp` | Declares the JSON runtime contract shared between Python setup and the C++ loop |
| `runtime_logging.h` | `../runtime_logging.cpp` | Defines the event/window/summary logging API used during gameplay |
| `runtime_timing.h` | `../runtime_timing.cpp` | Sample/time conversion helpers used across the runtime |
| `spike_detection.h` | `../spike_detection.cpp` | Streaming spike detector config and per-channel detector interface |
| `filter.h` | `../filter.cpp` | DSP primitives used by `SpikeDetector` |
| `gamewindow.h` | `../gamewindow.cpp` | Optional Qt visualization wrapper; compiles to a stub when `USE_QT=0` |

## Dependency Notes

| Module | Important dependencies |
| --- | --- |
| `maxone_with_filter.cpp` | Pulls in almost every header here and is the integration point for the full runtime |
| `spike_detection.h` | Depends on `filter.h`; filtering details should stay encapsulated there |
| `runtime_config.h` | Mirrors fields written by `pong_setup.py`; update both sides together |
| `runtime_logging.h` | Its CSV and JSONL schema is consumed by `analysis/behavior_metrics.py` and validation helpers |
| `ponggame.h` | Exposes only game-state behavior; it does not know about hardware or MaxLab |

## Editing Guidelines

| If you are changing... | Read these first |
| --- | --- |
| Feedback behavior or rally semantics | `ponggame.h`, `ponggame.cpp`, `tests/ponggame_test.cpp` |
| Baseline normalization or motor scaling | `motor_decoder.h`, `motor_decoder.cpp`, `tests/motor_decoder_test.cpp` |
| JSON config schema | `runtime_config.h`, `runtime_config.cpp`, `pong_setup.py`, `tests/runtime_config_test.cpp` |
| Log format | `runtime_logging.h`, `runtime_logging.cpp`, `analysis/behavior_metrics.py`, `tests/runtime_logging_test.cpp` |
| Spike-threshold behavior | `spike_detection.h`, `spike_detection.cpp`, `filter.h`, `filter.cpp` |
