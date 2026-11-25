"""Run the full native hydrogen curve and Bayesian phase-difference estimation."""

import argparse
from pathlib import Path

from . import model, plots, workflow


def report(result: workflow.Result) -> None:
    posterior = result.history[-1]
    scans = (
        result.check,
        result.diagnostic,
        *result.study,
        *(row.scan for row in result.history),
    )
    print(
        "FCI curve sizes:", (len(result.curve), len(model.differences(result.curve)[0]))
    )
    print("Fresh displaced FCI difference:", result.b.fci - result.a.fci)
    print(
        "Recorded study times and repetitions:",
        tuple((row.time, row.repetitions) for row in result.study),
    )
    print("Initial three observed Z estimates:", result.check.z_estimates)
    print("Observed 200-time Z scan:", result.diagnostic.z_estimates)
    print(
        "Bayesian posterior updates:",
        tuple(
            (row.posterior_mu, row.posterior_sigma, row.scan.repetitions)
            for row in result.history
        ),
    )
    print(
        "Posterior threshold reached:", posterior.posterior_sigma < workflow.THRESHOLD
    )
    print(
        "Final gradient and standard uncertainty:",
        (
            posterior.posterior_mu / model.DISPLACEMENT,
            posterior.posterior_sigma / model.DISPLACEMENT,
        ),
    )
    print(
        "Actual finite Bernoulli trials:",
        sum(len(row.energies) for row in scans) * workflow.SHOTS,
    )
    print("Maximum complete-state error:", max(row.maximum_error for row in scans))
    print(
        "Packed circuit operations per binding:",
        tuple(sorted({row.resources for row in scans})),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("bpde-results"))
    args = parser.parse_args()
    result = workflow.run()
    report(result)
    plots.save(result, args.figures)


if __name__ == "__main__":
    main()
