"""Prepare, submit, and evaluate a checkpointed regional Twin suite on Quafu.

Adjust the cell artifact paths and physical mapping. ``prepare`` is offline.
Each ``submit INDEX`` command creates exactly one task and refuses an existing
checkpoint. ``evaluate`` creates no task and requires every planned checkpoint.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import flagquantum as fq
from flagquantum.remote.qpu import QuafuProvider

PHYSICAL_QUBITS = (20, 27, 34)
SHOTS = 1024
REPETITIONS = 2
CELL_A_TWIN = Path("cell-a-twin.json")
CELL_A_SUPPORT = Path("cell-a-support.json")
CELL_B_TWIN = Path("cell-b-twin.json")
CELL_B_SUPPORT = Path("cell-b-support.json")
SUITE_PATH = Path("regional-twin-suite.json")
SUPPORT_PATH = Path("regional-twin-suite-support.json")
SUBMISSION_PREFIX = "regional-twin-submission"
SERIES_PREFIX = "regional-twin-series"


def circuits():
    """Return the exact ordered suite workload."""

    return (
        fq.Circuit(3).h(0).cx(0, 1).cx(1, 2),
        fq.Circuit(3).h(1).cx(1, 2),
        fq.Circuit(3).h(0).cx(0, 1).h(2),
    )


def prepare() -> None:
    """Freeze the suite without contacting a provider."""

    twin_a = fq.twin.load_twin(CELL_A_TWIN)
    support_a = fq.twin.load_circuit_support(CELL_A_SUPPORT)
    twin_b = fq.twin.load_twin(CELL_B_TWIN)
    support_b = fq.twin.load_circuit_support(CELL_B_SUPPORT)
    region_twin = fq.twin.compose_region_twin(
        ((twin_a, support_a), (twin_b, support_b))
    )
    suite = fq.twin.prepare_region_validation_suite(
        region_twin,
        circuits(),
        physical_qubits=PHYSICAL_QUBITS,
        name="flagquantum-regional-suite",
        shots=SHOTS,
        repetitions=REPETITIONS,
    )
    fq.twin.dump_region_validation_suite(suite, SUITE_PATH)
    print("planned tasks:", suite.planned_task_count)
    print("planned shots:", suite.planned_shots)
    print("suite:", SUITE_PATH)


def _submission_path(index: int) -> Path:
    return Path(f"{SUBMISSION_PREFIX}-{index:03d}.json")


def submit_one(provider: QuafuProvider, index: int) -> None:
    """Submit exactly one predeclared task and checkpoint its receipt."""

    suite = fq.twin.load_region_validation_suite(SUITE_PATH)
    if index < 1 or index > suite.planned_task_count:
        raise ValueError(f"INDEX must be in 1..{suite.planned_task_count}")
    destination = _submission_path(index)
    if destination.exists():
        raise FileExistsError(f"refusing to resubmit checkpointed task {index}")
    experiment_index = (index - 1) // suite.repetitions
    experiment = suite.experiments[experiment_index]
    receipt = experiment.submit(provider)
    submission = fq.twin.TwinSubmission.from_receipt(experiment, receipt)
    fq.twin.dump_submission(submission, destination)
    print("submitted task:", receipt.task_id)
    print("checkpoint:", destination)


def evaluate(provider: QuafuProvider) -> None:
    """Fetch and validate all existing tasks without submitting or retrying."""

    suite = fq.twin.load_region_validation_suite(SUITE_PATH)
    submissions = tuple(
        fq.twin.load_submission(_submission_path(index))
        for index in range(1, suite.planned_task_count + 1)
    )
    results = tuple(provider.fetch_result(item.receipt) for item in submissions)
    evaluation = suite.validate_results(
        submissions,
        results,
        circuits=circuits(),
        confidence_level=0.95,
    )
    support = evaluation.to_circuit_support()
    fq.twin.dump_circuit_support(support, SUPPORT_PATH)
    for index, series in enumerate(evaluation.validation_series, start=1):
        fq.twin.dump_validation_series(
            series,
            f"{SERIES_PREFIX}-{index:02d}.json",
        )

    print("circuits:", evaluation.circuit_count)
    print("tasks:", evaluation.task_count)
    print("shots:", evaluation.total_shots)
    print("Twin-QPU agreement:", f"{evaluation.mean_twin_qpu_agreement:.2%}")
    print("ideal-SV-QPU agreement:", f"{evaluation.mean_ideal_qpu_agreement:.2%}")
    print("QPU repeatability:", f"{evaluation.mean_qpu_repeatability:.2%}")
    print(
        "simultaneous shot radius:",
        f"{evaluation.simultaneous_finite_shot_tv_radius:.2%}",
    )
    print(
        "simultaneous TV error bound:",
        f"{evaluation.simultaneous_tv_error_bound:.2%}",
    )
    print("confidence level:", f"{evaluation.confidence_level:.2%}")
    print("support:", SUPPORT_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("prepare")
    submit = subcommands.add_parser("submit")
    submit.add_argument("index", type=int)
    subcommands.add_parser("evaluate")
    args = parser.parse_args()

    if args.command == "prepare":
        prepare()
        return
    if not os.getenv("QUAFU_API_TOKEN"):
        raise RuntimeError("Set QUAFU_API_TOKEN before contacting Quafu SQC")
    provider = QuafuProvider()
    if args.command == "submit":
        submit_one(provider, args.index)
    else:
        evaluate(provider)


if __name__ == "__main__":
    main()
