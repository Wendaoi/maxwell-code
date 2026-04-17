"""Behavior metrics derived from runtime event and window logs."""

from __future__ import annotations

import re
from typing import Iterable, Mapping


_BOUNCES_RE = re.compile(r"(?:^|,)bounces=(\d+)(?:,|$)")


def _event_bounces(event: Mapping[str, object]) -> int | None:
    match = _BOUNCES_RE.search(str(event.get("detail", "")))
    if match is None:
        return None
    return int(match.group(1))


def _window_elapsed_seconds(window: Mapping[str, object]) -> float:
    return float(window.get("elapsed_ms", 0)) / 1000.0


def _game_window_elapsed_seconds(
    windows: list[Mapping[str, object]],
) -> list[float]:
    raw_elapsed = [_window_elapsed_seconds(window) for window in windows]
    if len(raw_elapsed) <= 1:
        return raw_elapsed

    looks_cumulative = (
        any(value != raw_elapsed[0] for value in raw_elapsed[1:])
        and all(current >= previous for previous, current in zip(raw_elapsed, raw_elapsed[1:]))
    )
    if looks_cumulative:
        return raw_elapsed

    cumulative_elapsed: list[float] = []
    total = 0.0
    for value in raw_elapsed:
        total += value
        cumulative_elapsed.append(total)
    return cumulative_elapsed


def compute_behavior_metrics(
    windows: Iterable[Mapping[str, object]],
    events: Iterable[Mapping[str, object]],
    exclude_initial_game_seconds: int = 10,
) -> dict[str, object]:
    """Compute Figure 5/6 behavior metrics from replayable logs."""
    rally_lengths: list[int] = []
    hits_in_current_rally = 0
    for event in events:
        if event.get("phase") != "game":
            continue
        event_name = event.get("event")
        if event_name == "hit":
            hits_in_current_rally += 1
            continue
        if event_name != "miss":
            continue

        bounces = _event_bounces(event)
        if bounces is None:
            bounces = hits_in_current_rally
        elif bounces == 0 and hits_in_current_rally > 0:
            bounces = hits_in_current_rally

        rally_lengths.append(bounces)
        hits_in_current_rally = 0

    game_windows = [window for window in windows if window.get("phase") == "game"]
    game_elapsed_seconds = _game_window_elapsed_seconds(game_windows)
    kept_windows = [
        window
        for window, elapsed_seconds in zip(game_windows, game_elapsed_seconds)
        if elapsed_seconds >= exclude_initial_game_seconds
    ]

    paddle_movement = 0
    previous_y = None
    for window in kept_windows:
        paddle_y = int(float(window.get("paddle_y", 0)))
        if previous_y is not None:
            paddle_movement += abs(paddle_y - previous_y)
        previous_y = paddle_y

    average_rally_length = (
        sum(rally_lengths) / len(rally_lengths)
        if rally_lengths
        else 0.0
    )

    return {
        "rally_lengths": rally_lengths,
        "average_rally_length": average_rally_length,
        "aces": sum(1 for value in rally_lengths if value == 0),
        "long_rallies": sum(1 for value in rally_lengths if value > 3),
        "paddle_movement": paddle_movement,
    }
