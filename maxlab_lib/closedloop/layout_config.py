"""Paper-aligned electrode layout helpers for the Pong closed-loop experiment."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

Coordinate = Tuple[int, int]
Rect = Tuple[Coordinate, Coordinate]

ELECTRODE_COLUMNS = 220
ENTROPY_CLUSTER_COUNT = 18
ENTROPY_CLUSTER_SIZE = 50

SENSORY_STIM_COORDS: List[Coordinate] = [
    (20, 25),
    (40, 50),
    (20, 75),
    (40, 100),
    (20, 125),
    (40, 150),
    (20, 175),
    (40, 200),
]

MOTOR_DOWN_RECTS: List[Rect] = [
    ((80, 20), (109, 49)),
    ((80, 140), (109, 169)),
]

MOTOR_UP_RECTS: List[Rect] = [
    ((80, 50), (109, 79)),
    ((80, 170), (109, 199)),
]


def coord_to_electrode(row: int, col: int) -> int:
    """Convert a MaxOne grid coordinate to a routed electrode ID."""
    if row < 0 or col < 0:
        raise ValueError(f"Invalid MaxOne coordinate: row={row}, col={col}")
    if col >= ELECTRODE_COLUMNS:
        raise ValueError(f"Invalid MaxOne coordinate: row={row}, col={col}")
    return row * ELECTRODE_COLUMNS + col


def electrode_to_coord(electrode: int) -> Coordinate:
    """Convert a routed electrode ID back to a MaxOne grid coordinate."""
    if electrode < 0:
        raise ValueError(f"Invalid MaxOne electrode: {electrode}")
    return electrode // ELECTRODE_COLUMNS, electrode % ELECTRODE_COLUMNS


def sample_rect_1_in_4(top_left: Coordinate, bottom_right: Coordinate) -> List[int]:
    """Sample the upper-left electrode of each 2x2 block inside a rectangle."""
    row0, col0 = top_left
    row1, col1 = bottom_right
    if row1 < row0 or col1 < col0:
        raise ValueError(f"Invalid rectangle: {top_left} -> {bottom_right}")
    return [
        coord_to_electrode(row, col)
        for row in range(row0, row1 + 1, 2)
        for col in range(col0, col1 + 1, 2)
    ]


SENSORY_STIM_ELECTRODES = [coord_to_electrode(row, col) for row, col in SENSORY_STIM_COORDS]

MOTOR_DOWN_LEFT_ELECTRODES = sample_rect_1_in_4(*MOTOR_DOWN_RECTS[0])
MOTOR_DOWN_RIGHT_ELECTRODES = sample_rect_1_in_4(*MOTOR_DOWN_RECTS[1])
MOTOR_UP_LEFT_ELECTRODES = sample_rect_1_in_4(*MOTOR_UP_RECTS[0])
MOTOR_UP_RIGHT_ELECTRODES = sample_rect_1_in_4(*MOTOR_UP_RECTS[1])

MOTOR_DOWN_RECORDING_ELECTRODES = MOTOR_DOWN_LEFT_ELECTRODES + MOTOR_DOWN_RIGHT_ELECTRODES
MOTOR_UP_RECORDING_ELECTRODES = MOTOR_UP_LEFT_ELECTRODES + MOTOR_UP_RIGHT_ELECTRODES


def nearest_sensory_recording_electrodes(limit: int) -> List[int]:
    """Pick deterministic recording electrodes near the sensory stimulation area."""
    stim_set = set(SENSORY_STIM_ELECTRODES)
    motor_set = set(MOTOR_DOWN_RECORDING_ELECTRODES + MOTOR_UP_RECORDING_ELECTRODES)
    candidates: List[Tuple[float, int]] = []

    for row in range(0, 80):
        for col in range(0, ELECTRODE_COLUMNS):
            electrode = coord_to_electrode(row, col)
            if electrode in stim_set or electrode in motor_set:
                continue
            nearest_distance = min(
                (row - stim_row) ** 2 + (col - stim_col) ** 2
                for stim_row, stim_col in SENSORY_STIM_COORDS
            )
            candidates.append((nearest_distance, electrode))

    candidates.sort()
    return [electrode for _, electrode in candidates[:limit]]


SENSORY_RECORDING_ELECTRODES = nearest_sensory_recording_electrodes(124)
RECORDING_ELECTRODES = (
    MOTOR_DOWN_RECORDING_ELECTRODES +
    MOTOR_UP_RECORDING_ELECTRODES +
    SENSORY_RECORDING_ELECTRODES
)


def _region_for_electrode(electrode: int) -> str:
    if electrode in MOTOR_DOWN_LEFT_ELECTRODES:
        return "motor_1_down"
    if electrode in MOTOR_DOWN_RIGHT_ELECTRODES:
        return "motor_2_down"
    if electrode in MOTOR_UP_LEFT_ELECTRODES:
        return "motor_1_up"
    if electrode in MOTOR_UP_RIGHT_ELECTRODES:
        return "motor_2_up"
    if electrode in SENSORY_RECORDING_ELECTRODES:
        return "sensory"
    if electrode in SENSORY_STIM_ELECTRODES:
        return "stim"
    return "unassigned"


def _entropy_cluster_for_index(index: int) -> Optional[int]:
    if index < ENTROPY_CLUSTER_COUNT * ENTROPY_CLUSTER_SIZE:
        return index // ENTROPY_CLUSTER_SIZE
    return None


def build_electrode_metadata(
    channel_lookup: Dict[int, int],
    stim_unit_by_electrode: Dict[int, int],
) -> List[Dict[str, Optional[int] | str | bool]]:
    """Build per-electrode metadata needed to replay and analyze a session."""
    metadata: List[Dict[str, Optional[int] | str | bool]] = []
    sorted_recording = sorted(
        RECORDING_ELECTRODES,
        key=lambda electrode: electrode_to_coord(electrode),
    )

    for index, electrode in enumerate(sorted_recording):
        row, col = electrode_to_coord(electrode)
        metadata.append({
            "electrode": electrode,
            "channel": channel_lookup.get(electrode),
            "row": row,
            "col": col,
            "region": _region_for_electrode(electrode),
            "entropy_cluster": _entropy_cluster_for_index(index),
            "is_stim": electrode in stim_unit_by_electrode,
            "stim_unit": stim_unit_by_electrode.get(electrode),
        })

    return metadata

assert len(SENSORY_STIM_ELECTRODES) == 8
assert len(MOTOR_DOWN_LEFT_ELECTRODES) == 225
assert len(MOTOR_DOWN_RIGHT_ELECTRODES) == 225
assert len(MOTOR_UP_LEFT_ELECTRODES) == 225
assert len(MOTOR_UP_RIGHT_ELECTRODES) == 225
assert len(MOTOR_DOWN_RECORDING_ELECTRODES) == 450
assert len(MOTOR_UP_RECORDING_ELECTRODES) == 450
assert len(SENSORY_RECORDING_ELECTRODES) == 124
assert len(RECORDING_ELECTRODES) == 1024
assert len(set(RECORDING_ELECTRODES + SENSORY_STIM_ELECTRODES)) == 1032
