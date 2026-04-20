# closedloop

`closedloop/` is the implementation root for the Pong closed-loop experiment. It contains the Python setup/orchestration layer, the C++ runtime that turns spike activity into paddle movement, the post-hoc analysis helpers, and the regression tests around that workflow.

## Directory Map

| Path | Role |
| --- | --- |
| `pong_setup.py` | Main experiment entrypoint. Initializes MaxLab, prepares stimulation sequences, writes session metadata/config, starts recording, launches the C++ runtime, and performs lightweight post-run inspection. |
| `layout_config.py` | Paper-aligned electrode layout definition: 8 sensory stimulation electrodes, 900 motor recording electrodes, 124 sensory recording electrodes, and entropy-cluster metadata. |
| `runtime_args.py` | Normalizes CLI condition names and builds the minimal argv contract consumed by the C++ executable. |
| `maxone_with_filter.cpp` | Main runtime loop. Loads JSON config, receives streamed data, decodes motor activity, advances Pong, emits stimulation, and logs events/windows/quality data. |
| `ponggame.cpp` | Pure game-state update logic and the condition-specific hit/miss behavior. |
| `motor_decoder.cpp` | Baseline-aware gain normalization for up/down motor channel groups. |
| `spike_detection.cpp` | Streaming spike detector for raw traces, including 100 Hz high-pass activity extraction, 1 Hz MAD estimation, a separate 1 Hz median side path, and refractory handling. |
| `filter.cpp` | First-order low-pass and second-order high-pass filters used by `spike_detection.cpp`. |
| `runtime_config.cpp` | JSON parser and runtime contract loader for the Python-to-C++ handoff. |
| `runtime_logging.cpp` | Writes `runtime_events.jsonl`, `window_samples.csv`, and `quality_summary.json`. |
| `runtime_timing.cpp` | Centralized sample-count and elapsed-time calculations. |
| `gamewindow.cpp` | Optional Qt visualizer for the paddle and ball state. |
| `include/` | Public headers for the C++ closed-loop runtime. |
| `analysis/` | Post-hoc helpers for behavior metrics, daily scan manifests, entropy/CA metrics, and session validation. |
| `tests/` | C++ and Python regression tests for the runtime contract and analysis helpers. |

## Runtime Flow

1. `pong_setup.py` normalizes the requested condition, prepares session metadata, and selects/routs recording plus stimulation electrodes.
2. The setup script pre-generates stimulation sequences:
   - `80` ball-position sequences (`8 positions x 10 frequencies`)
   - `1` hit feedback sequence
   - `8` miss feedback variants
3. Python writes `session_manifest.json`, `session_config.json`, and `resolved_layout.json`, then starts MaxLab recording and launches `build/maxone_with_filter`.
4. During the pre-rest phase, the C++ runtime accumulates motor baseline windows and freezes decoder gains before gameplay begins.
5. During gameplay, the runtime advances the Pong model every `10 ms`, chooses the sensory sequence that matches current ball position and horizontal velocity bin, emits hit/miss feedback according to the selected condition, and applies readout-only blinding after sensory and feedback stimulation so artifacts do not enter the motor decoder.
6. The runtime logs replayable events plus per-window motor/game state, and the `analysis/` helpers consume those files later for figure-level metrics.

## Supported Runtime Conditions

Only the feedback logic lives in the runtime condition enum:

| Condition | Behavior in code |
| --- | --- |
| `STIM` | Full experiment condition: sensory input + predictable hit reward + miss punishment |
| `SILENT` | Hit reward removed, miss replaced by equally long silent pause |
| `NO_FEEDBACK` | Sensory input only; no miss reset feedback |
| `REST` | No sensory stimulation; motor activity is still logged |

Substrate-level controls such as `CTL` and `HEK` are tracked as metadata (`--cell-type`) rather than as separate runtime conditions. The `IS` control from the schedule is not implemented as a first-class runtime mode in the current code.

## Files Written Per Session

Each run creates a session directory under `~/pong_experiments` by default:

| File | Meaning |
| --- | --- |
| `session_manifest.json` | Human-oriented summary: culture metadata, phase durations, analysis defaults, output paths |
| `session_config.json` | Full machine-readable runtime config passed to C++ |
| `resolved_layout.json` | Routed recording/stimulation electrodes and per-electrode metadata |
| `runtime_events.jsonl` | Phase transitions, stimulation dispatches, hits, misses |
| `window_samples.csv` | Motor counts, gains, ball state, paddle state, sensory bin, rally id per window |
| `quality_summary.json` | Total / accepted / corrupted / dropped frame counts |
| `*.raw.h5` | Raw MaxLab recording |

## Where To Read Next

- `include/README.md`: header ownership and the C++ API surface
- `analysis/README.md`: post-hoc analysis commands and metric helpers
- `tests/README.md`: regression suite and recommended verification commands
