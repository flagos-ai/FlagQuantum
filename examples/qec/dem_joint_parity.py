"""Show how one shared DEM fault correlates detector and logical parity."""

from __future__ import annotations

import json
from itertools import product

from flagquantum.qec import DetectorErrorModel


def main() -> None:
    """Print the four joint probabilities for one fixed two-mechanism DEM."""
    model = DetectorErrorModel.from_stim_text(
        "error(0.125) D0 L0\nerror(0.25) D0\ndetector D0\nlogical_observable L0\n"
    )
    probabilities = dict.fromkeys(("00", "01", "10", "11"), 0.0)
    for fires in product((False, True), repeat=2):
        detector = logical = False
        probability = 1.0
        for fired, error, rate in zip(
            fires, model.errors, model.error_rates, strict=True
        ):
            probability *= rate if fired else 1.0 - rate
            if fired:
                detector ^= 0 in error.detectors
                logical ^= 0 in error.observables
        probabilities[f"{int(detector)}{int(logical)}"] += probability

    detector_rate = probabilities["10"] + probabilities["11"]
    logical_rate = probabilities["01"] + probabilities["11"]
    if (
        sum(probabilities.values()) != 1.0
        or detector_rate != model.detector_rates().item()
        or logical_rate != model.observable_rates().item()
    ):
        raise RuntimeError("joint parity probabilities disagree with the fixed DEM")
    print(json.dumps(probabilities))


if __name__ == "__main__":
    main()
