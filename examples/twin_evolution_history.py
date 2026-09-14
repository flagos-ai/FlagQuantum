"""Align saved QPU calibration drift with Twin validation changes offline."""

from __future__ import annotations

import argparse
from pathlib import Path

import flagquantum as fq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-history", type=Path, required=True)
    parser.add_argument("--validation-history", type=Path, required=True)
    arguments = parser.parse_args()

    calibration = fq.twin.load_calibration_history(arguments.calibration_history)
    validation = fq.twin.load_validation_history(arguments.validation_history)
    evolution = fq.twin.align_histories(calibration, validation)
    drift = evolution.latest_calibration_drift

    print(f"{calibration.provider}:{calibration.backend_name}")
    print("physical-qubits", calibration.physical_qubits)
    print("observations", evolution.observation_count)
    print("latest-device-drift", drift.has_observed_drift)
    print(f"Twin-QPU-change={evolution.latest_twin_agreement_change:+.2%}")
    print(f"Ideal-QPU-change={evolution.latest_ideal_agreement_change:+.2%}")
    repeatability = evolution.latest_qpu_repeatability_change
    print(
        "QPU-repeatability-change=unavailable"
        if repeatability is None
        else f"QPU-repeatability-change={repeatability:+.2%}"
    )
    print(f"verified-TV-bound-change={evolution.latest_verified_bound_change:+.2%}")


if __name__ == "__main__":
    main()
