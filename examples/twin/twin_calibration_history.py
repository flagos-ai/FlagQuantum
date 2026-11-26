"""Print drift history from two or more saved QPU Twin model artifacts.

This example is offline. Create each input first with ``fq.twin.dump_twin``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import flagquantum as fq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "twins",
        nargs="+",
        type=Path,
        help="Twin JSON files in strictly increasing calibration time",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optionally save the canonical calibration-history JSON",
    )
    arguments = parser.parse_args()
    paths = arguments.twins
    if len(paths) < 2:
        parser.error("provide at least two saved Twin files")

    history = fq.twin.build_calibration_history(
        [fq.twin.load_twin(path) for path in paths]
    )
    if arguments.output is not None:
        fq.twin.dump_calibration_history(history, arguments.output)
    print(
        f"{history.provider}:{history.backend_name}",
        history.physical_qubits,
    )
    for timestamp, cumulative, incremental in zip(
        history.captured_at[1:],
        history.baseline_drifts,
        history.interval_drifts,
    ):
        print(
            timestamp,
            f"T1 from baseline={cumulative.maximum_relative_t1_change:.2%}",
            f"T1 since previous={incremental.maximum_relative_t1_change:.2%}",
        )


if __name__ == "__main__":
    main()
