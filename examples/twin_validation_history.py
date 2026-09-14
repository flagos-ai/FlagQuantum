"""Print longitudinal accuracy metrics from frozen Twin validation artifacts.

This example is offline. Each ``--observation`` pairs one saved Twin model with
the validation series produced prospectively for that exact snapshot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import flagquantum as fq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--observation",
        action="append",
        nargs=2,
        metavar=("TWIN_JSON", "VALIDATION_JSON"),
        type=Path,
        required=True,
        help="repeat in strictly increasing calibration time",
    )
    arguments = parser.parse_args()
    if len(arguments.observation) < 2:
        parser.error("provide at least two --observation pairs")

    observations = [
        (
            fq.twin.load_twin(twin_path),
            fq.twin.load_validation_series(validation_path),
        )
        for twin_path, validation_path in arguments.observation
    ]
    history = fq.twin.build_validation_history(observations)

    print(
        f"{history.provider}:{history.backend_name}",
        history.physical_qubits,
    )
    for timestamp, twin_agreement, ideal_agreement, repeatability, radius, bound in zip(
        history.captured_at,
        history.mean_twin_qpu_agreements,
        history.mean_ideal_qpu_agreements,
        history.mean_qpu_repeatabilities,
        history.simultaneous_finite_shot_tv_radii,
        history.verified_tv_error_bounds,
        strict=True,
    ):
        repeatability_text = (
            "unavailable" if repeatability is None else f"{repeatability:.2%}"
        )
        print(
            timestamp,
            f"Twin-QPU={twin_agreement:.2%}",
            f"Ideal-QPU={ideal_agreement:.2%}",
            f"QPU-repeatability={repeatability_text}",
            f"shot-radius={radius:.2%}",
            f"verified-TV-bound={bound:.2%}",
        )


if __name__ == "__main__":
    main()
