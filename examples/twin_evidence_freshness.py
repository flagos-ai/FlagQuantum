"""Assess persisted Twin evidence against explicit application thresholds."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import flagquantum as fq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("twin", help="Path to the current frozen Twin JSON")
    parser.add_argument("calibration_history", help="Calibration-history JSON")
    parser.add_argument("validation_history", help="Validation-history JSON")
    args = parser.parse_args()

    twin = fq.twin.load_twin(args.twin)
    calibration_history = fq.twin.load_calibration_history(args.calibration_history)
    validation_history = fq.twin.load_validation_history(args.validation_history)

    policy = fq.twin.TwinFreshnessPolicy(
        maximum_calibration_age=timedelta(hours=4),
        maximum_validation_age=timedelta(hours=8),
        maximum_relative_t1_drift=0.20,
        maximum_relative_t2_drift=0.20,
        maximum_readout_tv_drift=0.02,
        maximum_relative_gate_duration_drift=0.10,
        maximum_verified_tv_error_bound=0.10,
    )
    assessment = policy.assess(
        twin,
        calibration_history=calibration_history,
        validation_history=validation_history,
        as_of=datetime.now(timezone.utc),
    )

    print(f"status: {assessment.status}")
    print(f"reasons: {assessment.reasons}")
    print(f"next refresh: {assessment.next_refresh_at}")


if __name__ == "__main__":
    main()
