"""Spike extraction helpers for post-hoc raw trace analysis."""

from __future__ import annotations

from typing import Iterable, Mapping


def detect_threshold_crossings(
    traces: Mapping[int, Iterable[float]],
    threshold: float,
    refractory_samples: int,
) -> list[dict[str, float | int]]:
    """Detect negative-going threshold crossings in channel traces."""
    spikes: list[dict[str, float | int]] = []
    refractory_samples = max(1, int(refractory_samples))

    for channel, values in traces.items():
        previous = 0.0
        samples_since_spike = refractory_samples
        for sample, value in enumerate(values):
            value = float(value)
            if samples_since_spike >= refractory_samples and previous > threshold >= value:
                spikes.append({
                    "channel": int(channel),
                    "sample": sample,
                    "amplitude": value,
                })
                samples_since_spike = 0
            else:
                samples_since_spike += 1
            previous = value

    return spikes
