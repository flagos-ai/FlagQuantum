"""Read the original CSV and the Notebook's separately published Qiskit rows."""

import csv
from dataclasses import dataclass
from pathlib import Path

SELECTED = (0, 5, 10, 12, 14, *range(15, 23))
FILE = Path(__file__).parent / "data" / "results.csv"


@dataclass(frozen=True)
class Record:
    position_bits: int
    width: int
    cx: int
    seconds: float


QISKIT = tuple(
    Record(bits, width, cx, seconds)
    for bits, width, cx, seconds in zip(
        range(5, 11),
        (6, 7, 8, 9, 10, 11),
        (900, 2376, 5388, 11472, 23700, 48216),
        (
            0.43783092498779297,
            3.743027925491333,
            17.858744144439697,
            73.1058611869812,
            293.51222825050354,
            1222.052798986435,
        ),
        strict=True,
    )
)


def read(path: Path = FILE) -> tuple[tuple[Record, ...], tuple[Record, ...]]:
    """Read the same thirteen rows and two separate source series."""
    with path.open(newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    records = []
    for suffix in ("width_opt", "cx_opt"):
        records.append(
            tuple(
                Record(
                    int(rows[row][f"log2_circle_size_{suffix}"]),
                    int(rows[row][f"width_{suffix}"]),
                    int(rows[row][f"cx_{suffix}"]),
                    float(rows[row][f"time_{suffix}"]),
                )
                for row in SELECTED
            )
        )
    return records[0], records[1]
