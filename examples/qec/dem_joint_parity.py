"""Show how one shared DEM fault correlates detector and logical parity."""

from __future__ import annotations

import argparse
import json
from itertools import product

from flagquantum.qec import DemError, DetectorErrorModel


def main(*, exclusive: bool = False) -> None:
    """Print the four joint probabilities for one fixed two-mechanism DEM."""
    masks: tuple[tuple[bool, ...], ...]
    if exclusive:
        model = DetectorErrorModel(
            num_detectors=1,
            num_observables=1,
            errors=(
                DemError(0.125, detectors=(0,), observables=(0,), error_id=0),
                DemError(0.25, detectors=(0,), error_id=0),
            ),
        )
        masks = ((False, False), (True, False), (False, True))
    else:
        model = DetectorErrorModel.from_stim_text(
            "error(0.125) D0 L0\nerror(0.25) D0\ndetector D0\nlogical_observable L0\n"
        )
        masks = tuple(product((False, True), repeat=2))

    probabilities = dict.fromkeys(("00", "01", "10", "11"), 0.0)
    rates = model.error_rates
    for fires in masks:
        detector = logical = False
        probability = 1.0
        for fired, error, rate in zip(fires, model.errors, rates, strict=True):
            if not exclusive:
                probability *= rate if fired else 1.0 - rate
            if fired:
                detector ^= 0 in error.detectors
                logical ^= 0 in error.observables
        if exclusive:
            probability = rates[fires.index(True)] if any(fires) else 1.0 - sum(rates)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exclusive", action="store_true", help="Use mutually exclusive faults"
    )
    main(exclusive=parser.parse_args().exclusive)
