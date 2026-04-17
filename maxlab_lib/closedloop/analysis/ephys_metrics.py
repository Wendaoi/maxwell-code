"""Electrophysiology metrics that can be computed after recording."""

from __future__ import annotations

import math
from typing import Iterable, Mapping


def binary_entropy(p: float) -> float:
    """Return Bernoulli entropy in shannons."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p)) - ((1.0 - p) * math.log2(1.0 - p))


def cross_correlation(left: Iterable[float], right: Iterable[float]) -> float:
    """Pearson correlation for matched 100 ms spike-count bins."""
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if len(left_values) != len(right_values) or not left_values:
        raise ValueError("correlation inputs must be non-empty and equal length")

    left_mean = sum(left_values) / len(left_values)
    right_mean = sum(right_values) / len(right_values)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left_values, right_values)
    )
    left_var = sum((x - left_mean) ** 2 for x in left_values)
    right_var = sum((y - right_mean) ** 2 for y in right_values)
    denominator = math.sqrt(left_var * right_var)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def cluster_binary_entropy(spike_presence: Iterable[int]) -> float:
    """Binary entropy for one spatial cluster in one 100 ms bin."""
    values = [1 if value else 0 for value in spike_presence]
    if not values:
        return 0.0
    return binary_entropy(sum(values) / len(values))


def center_of_activity(samples: Iterable[Mapping[str, float]]) -> tuple[float, float]:
    """Compute CA from per-electrode spike counts and coordinates."""
    total_spikes = 0.0
    weighted_x = 0.0
    weighted_y = 0.0

    for sample in samples:
        spikes = float(sample.get("spikes", 0.0))
        total_spikes += spikes
        weighted_x += spikes * float(sample.get("x", sample.get("col", 0.0)))
        weighted_y += spikes * float(sample.get("y", sample.get("row", 0.0)))

    if total_spikes == 0.0:
        return (0.0, 0.0)
    return (weighted_x / total_spikes, weighted_y / total_spikes)


def functional_plasticity_distance(
    ca: tuple[float, float],
    rest_centroid: tuple[float, float],
) -> float:
    """Euclidean CA distance from the matched 10 min rest centroid."""
    return math.hypot(ca[0] - rest_centroid[0], ca[1] - rest_centroid[1])


def exclusive_motor_event_percentage(bins: Iterable[Mapping[str, float]]) -> float:
    """Percent of 1000 ms bins with activity in exactly one motor region."""
    total = 0
    exclusive = 0
    for bin_row in bins:
        motor_1_active = float(bin_row.get("motor_1_spikes", 0.0)) > 0.0
        motor_2_active = float(bin_row.get("motor_2_spikes", 0.0)) > 0.0
        total += 1
        if motor_1_active != motor_2_active:
            exclusive += 1

    if total == 0:
        return 0.0
    return exclusive * 100.0 / total
