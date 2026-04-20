# maxwell-code

## Build

Build the closed-loop C++ executable from the MaxLab library directory:

```bash
cd maxwell-code/maxlab_lib
make maxone_with_filter USE_QT=0
```

Use `USE_QT=1` only on systems where the Qt6 headers and libraries are installed and discoverable by the compiler/linker. The headless `USE_QT=0` build is the default path for the Python-led experiment flow.

## Run

Start one closed-loop session from Python:

```bash
cd maxwell-code/maxlab_lib/closedloop
python3 pong_setup.py --duration 20 --condition STIM --wells 0
```

`pong_setup.py` owns the startup sequence. It initializes MaxLab, configures routing, connects stimulation units, prepares named stimulation sequences, exports the JSON runtime config, starts raw HDF5 recording, launches the C++ executable with the config path, records a baseline, then sends `start` to the C++ process.

The current runtime contract is single-well. Pass exactly one well, for example `--wells 0`; multi-well runs are rejected before hardware setup because the C++ runtime consumes one routed-channel map.

Raw voltage streaming is the default closed-loop mode. The JSON config carries `runtime.stream_mode`, and the C++ runtime keeps a filtered-stream fallback for setups that explicitly select it.

## CLI Quick Reference

The flags below are the parameters you will actually touch during an experiment run:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--duration` | `20` | Gameplay duration in minutes. The paper-aligned default is 20 min. |
| `--pre-rest-seconds` | `600` | Baseline/rest recording before gameplay. Set `0` to skip. |
| `--condition` | `STIM` | Feedback logic. Supported values: `STIM`, `STIMULUS`, `SILENT`, `NO_FEEDBACK`, `NO-FEEDBACK`, `REST`. |
| `--wells` | `0` | Target well index. Current runtime supports exactly one well. |
| `--culture-id` | `unknown_culture` | Human-readable culture identifier, used in session naming. |
| `--cell-type` | `unknown` | Cell type tag, e.g. `MCC`, `HCC_DSI`, `HCC_NGN2`, `HEK`, `CTL`. |
| `--replicate-id` | `unknown_replicate` | Chip / biological replicate identifier. |
| `--experiment-day` | `0` | Day index written into `session_manifest.json`. |
| `--session-index` | `1` | Session number within the same test day. |
| `--operator` | empty | Operator initials / name. |
| `--notes` | empty | Free-text notes for later traceability. |
| `--recording-root` | `~/pong_experiments` | Root directory for generated session folders. |

Recommended session command template:

```bash
cd maxwell-code/maxlab_lib/closedloop
python3 pong_setup.py \
  --duration 20 \
  --pre-rest-seconds 600 \
  --condition STIM \
  --wells 0 \
  --culture-id culture_a \
  --cell-type MCC \
  --replicate-id chip01 \
  --experiment-day 14 \
  --session-index 1 \
  --operator wd \
  --notes "day1 session1"
```

## Experiment Quick Reference

This section summarizes the parameters from `Pong game 实验复现日程表.md` and aligns them with the current codebase so they can be checked quickly during wet-lab execution.

### 1. Cell Preparation And Culture

| Cell source | Notes from the schedule | Maturation / test window |
| --- | --- | --- |
| `MCC` | Mouse primary cortical culture from E15.5 embryos | Roughly 14 days to mature |
| `HCC_DSI` | Human cortical cells prepared via dual-SMAD inhibition | Roughly 73 days to mature |
| `HCC_NGN2` | NGN2 lentiviral direct reprogramming | Roughly 14-24 days to mature |
| `HEK` | Non-neuronal control | Can be tested about 24 h after plating |
| `CTL` | Medium-only control, no cells | No maturation period |

Common plating and culture parameters:

| Item | Value |
| --- | --- |
| Chip | MaxOne HD-MEA |
| Coating for primary culture | PEI |
| Coating for iPSC-derived culture | PDL |
| Shared extracellular coating | Laminin `10 ug/ml` |
| Seeding density | About `1,000,000` cells per chip |
| Incubator | `36 C`, `5% CO2`, `5% O2`, `80%` relative humidity |
| Medium change cadence | Half-volume change every 2 days |
| Operational note | Always complete electrophysiology recording before medium exchange |

### 2. Daily Baseline Scan For Figure 3

These parameters are not set by `pong_setup.py`; they are the daily checkerboard acquisition settings that should stay fixed across maturation tracking:

| Item | Value |
| --- | --- |
| Scan type | Checkerboard scan |
| Configurations per day | `14` |
| Record time per configuration | `15 s` |
| High-pass filter | `300 Hz` |
| Gain | `512x` |
| Daily activity-scan spike threshold | `6 sigma` above background noise |
| Closed-loop runtime spike detector | `100 Hz` 2nd-order Bessel high-pass -> absolute value -> `1 Hz` 1st-order Bessel low-pass MAD estimate -> MAD-scaled negative threshold crossing |
| Daily outputs to track | Mean firing rate, max firing rate, firing rate variance |

The helper in `maxlab_lib/closedloop/analysis/daily_activity_scan.py` writes these same fixed assumptions into `daily_activity_scan_manifest.json`.

### 3. Core Closed-Loop Session Schedule

Paper-aligned session structure:

| Phase | Duration | Notes |
| --- | --- | --- |
| Pre-rest | `10 min` (`600 s`) | Record spontaneous control activity before stimulation. |
| Gameplay | `20 min` (`1200 s`) | Closed-loop Pong session. |
| Analysis exclusion | First `10 s` of gameplay | Drop initialization-heavy period during post-hoc analysis. |
| Analysis windows | `T1 = 0-5 min`, `T2 = 6-20 min` | Stored in `session_manifest.json` as analysis defaults. |
| Sessions per day | `3` | Run 3 sessions per chip per day for 3 consecutive days. |

The current code defaults already match this schedule:

| Runtime field | Value in code |
| --- | --- |
| `pre_rest_seconds` | `600` |
| `game_seconds` | `1200` when `--duration 20` |
| `exclude_initial_game_seconds` | `10` |
| `window_ms` | `10` |
| `sample_rate_hz` | `20000` |

### 4. Experimental Groups And Current Code Mapping

The schedule defines 7 parallel groups, but the runtime currently implements 4 feedback-logic conditions directly. The remaining groups are represented by the biological substrate or by an external control setup.

| Schedule group | Meaning | Current code mapping |
| --- | --- | --- |
| Stimulus | Predictable hit reward + strong miss feedback | `--condition STIM` |
| Silent | No hit reward; miss replaced by equally long silence | `--condition SILENT` |
| No-feedback | Only position encoding; no reset / no miss feedback | `--condition NO_FEEDBACK` |
| RST | Rest control, no sensory input | `--condition REST` |
| CTL | Medium-only control, same feedback logic as Stimulus | Use `--condition STIM --cell-type CTL` |
| HEK | Non-neuronal live-cell control, same feedback logic as Stimulus | Use `--condition STIM --cell-type HEK` |
| IS | In-silico random controller | Not a built-in `pong_setup.py` condition in the current codebase; needs an external paddle driver / simulator layer |

### 5. Hardcoded Stimulation And Runtime Parameters

These are the parameters currently embedded in `pong_setup.py`, `layout_config.py`, and the C++ runtime:

| Category | Current implementation |
| --- | --- |
| Ball-position stimulation electrodes | `8` sensory stimulation electrodes |
| Ball-position frequencies | `4, 8, 12, 16, 20, 24, 28, 32, 36, 40 Hz` |
| Ball-position amplitude | `75 mV` |
| Ball-position phase width | `200 us` per phase |
| Ball-position sequences | `8 positions x 10 frequencies = 80` pre-generated sequences |
| Hit feedback | Predictable synchronous stimulation on all 8 sensory electrodes |
| Hit feedback params | `100 Hz`, `75 mV`, `100 ms`, `200 us` phase width |
| Miss feedback | Random sensory-electrode stimulation with 8 pre-generated variants |
| Miss feedback params | `5 Hz`, `150 mV`, `4 s`, `200 us` phase width |
| Miss pause after feedback | `4000 ms` |
| Hit sensory suppression | `100 ms` |
| Sensory readout blinding | `5 ms` |
| Hit readout blinding | `105 ms` (`100 ms` burst + `5 ms`) |
| Miss readout blinding | `4005 ms` (`4000 ms` burst + `5 ms`) |
| Motor decoder gain target | `20 Hz` |
| Stim-neighbor fallback radius | `2` electrode rings |
| Recording layout | `1024` recording electrodes total |
| Motor channels | `900` electrodes (`450 up`, `450 down`) |
| Sensory recording channels | `124` electrodes |
| Sensory stimulation channels | `8` electrodes |
| Entropy layout | `18` clusters x `50` electrodes |

### 6. Session Outputs And Figure-Oriented Metrics

Each session folder under `~/pong_experiments/<session_id>/` contains:

| File | Purpose |
| --- | --- |
| `session_manifest.json` | Human-readable metadata, phase durations, and analysis defaults |
| `session_config.json` | Full runtime contract passed from Python to C++ |
| `resolved_layout.json` | Final routed channel/electrode/stimulation layout used in the run |
| `runtime_events.jsonl` | Hit / miss / phase / stimulation events |
| `window_samples.csv` | Per-window motor/game state log |
| `quality_summary.json` | Corrupted / dropped / accepted frame summary |
| `*.raw.h5` | Raw recording file from MaxLab |

Metrics expected by the schedule and already supported by the analysis helpers:

| Figure family | Metric |
| --- | --- |
| Figure 5 / 6 / S4 | Average Rally Length |
| Figure 5 / 6 / S4 | Aces |
| Figure 5 / 6 / S4 | Long Rallies |
| Figure 5 / 6 / S4 | Paddle Movement |
| Figure 7 | Cross-correlation |
| Figure 7 | Exclusive motor region activity |
| Figure 7 | Functional plasticity (CA distance) |
| Figure 7 | Information entropy / cluster entropy helpers |

For module-level details, see:

- `maxlab_lib/closedloop/README.md`
- `maxlab_lib/closedloop/include/README.md`
- `maxlab_lib/closedloop/analysis/README.md`
- `maxlab_lib/closedloop/tests/README.md`

## Platform Notes

Install and activate the `maxlab` Python package before running `pong_setup.py`. MaxLab Live and the MaxLab C++/Python libraries must be available on the same machine that controls the hardware.

On Linux, MaxLab binaries can be sensitive to the active C++ runtime. If import or launch errors mention `libstdc++` or `GLIBCXX`, run from the environment recommended by MaxWell/MaxLab or adjust `LD_LIBRARY_PATH` so the MaxLab-compatible runtime is used.

The Makefile automatically adds `$(HOME)/MaxLab/lib` when it contains `libstdc++.so`, which matches the MaxLab Linux install layout and resolves newer `libstdc++` symbols required by the bundled `libmaxlab.a`. Override with `MAXLAB_RUNTIME_LIB=/path/to/MaxLab/lib` if your installation lives elsewhere.
