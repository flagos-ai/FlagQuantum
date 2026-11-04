"""Show duplicate mechanism consolidation and its independence requirement."""

import json

from flagquantum.qec import DemError, DetectorErrorModel


def main() -> None:
    """Report the exact merge and the refusal for an exclusive group."""
    independent = DetectorErrorModel(
        num_detectors=1,
        num_observables=1,
        errors=(DemError(0.125, (0,), (0,)), DemError(0.25, (0,), (0,))),
    )
    merged = independent.merge_duplicate_mechanisms()
    if (
        merged.num_errors != 1
        or merged.detector_rates().item() != independent.detector_rates().item()
        or merged.observable_rates().item() != independent.observable_rates().item()
    ):
        raise RuntimeError("the exact merge must preserve both marginal rates")

    exclusive = DetectorErrorModel(
        num_detectors=1,
        num_observables=1,
        errors=(
            DemError(0.125, (0,), (0,), error_id=0),
            DemError(0.25, (0,), (0,), error_id=0),
        ),
    )
    try:
        exclusive.merge_duplicate_mechanisms()
    except ValueError as error:
        if "error id" not in str(error):
            raise
    else:
        raise RuntimeError("the independent merge accepted an exclusive group")

    print(
        json.dumps(
            {
                "independent_count": merged.num_errors,
                "independent_probability": merged.error_rates[0],
                "exclusive_detector_rate": exclusive.detector_rates().item(),
                "exclusive_merge": "rejected",
            }
        )
    )


if __name__ == "__main__":
    main()
