"""Prepare, submit, and evaluate a regional Twin holdout study on Quafu.

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
STUDY_PATH = Path("regional-twin-holdout-study.json")
REFERENCE_SUPPORT_PATH = Path("regional-twin-reference-support.json")
HOLDOUT_SUPPORT_PATH = Path("regional-twin-holdout-support.json")
SUBMISSION_PREFIX = "regional-twin-holdout-submission"


def reference_circuits() -> tuple[fq.Circuit, ...]:
    """Return the ordered reference circuit group."""

    return (
        fq.Circuit(3).h(0).cx(0, 1).cx(1, 2),
        fq.Circuit(3).h(1).cx(1, 2),
    )


def holdout_circuits() -> tuple[fq.Circuit, ...]:
    """Return circuits frozen before reference results are available."""

    return (
        fq.Circuit(3).h(0).cx(0, 1),
        fq.Circuit(3).h(0).cx(0, 1).cx(1, 2).h(1),
    )


def prepare() -> None:
    """Freeze the split and every prediction without provider access."""

    twin_a = fq.twin.load_twin(CELL_A_TWIN)
    support_a = fq.twin.load_circuit_support(CELL_A_SUPPORT)
    twin_b = fq.twin.load_twin(CELL_B_TWIN)
    support_b = fq.twin.load_circuit_support(CELL_B_SUPPORT)
    region_twin = fq.twin.compose_region_twin(
        ((twin_a, support_a), (twin_b, support_b))
    )
    study = fq.twin.prepare_region_holdout_study(
        region_twin,
        reference_circuits(),
        holdout_circuits(),
        physical_qubits=PHYSICAL_QUBITS,
        name="flagquantum-regional-holdout",
        shots=SHOTS,
        repetitions=REPETITIONS,
    )
    fq.twin.dump_region_holdout_study(study, STUDY_PATH)
    print("reference circuits:", study.reference_circuit_count)
    print("holdout circuits:", study.holdout_circuit_count)
    print("planned tasks:", study.planned_task_count)
    print("planned shots:", study.planned_shots)
    print("study:", STUDY_PATH)


def _submission_path(index: int) -> Path:
    return Path(f"{SUBMISSION_PREFIX}-{index:03d}.json")


def _experiments(
    study: fq.twin.TwinRegionHoldoutStudy,
) -> tuple[fq.twin.TwinExperiment, ...]:
    return (*study.reference_suite.experiments, *study.holdout_suite.experiments)


def submit_one(provider: QuafuProvider, index: int) -> None:
    """Submit exactly one frozen task and checkpoint its receipt."""

    study = fq.twin.load_region_holdout_study(STUDY_PATH)
    if index < 1 or index > study.planned_task_count:
        raise ValueError(f"INDEX must be in 1..{study.planned_task_count}")
    destination = _submission_path(index)
    if destination.exists():
        raise FileExistsError(f"refusing to resubmit checkpointed task {index}")
    experiment_index = (index - 1) // study.reference_suite.repetitions
    experiment = _experiments(study)[experiment_index]
    receipt = experiment.submit(provider)
    submission = fq.twin.TwinSubmission.from_receipt(experiment, receipt)
    fq.twin.dump_submission(submission, destination)
    print("submitted task:", receipt.task_id)
    print("checkpoint:", destination)


def evaluate(provider: QuafuProvider) -> None:
    """Fetch and validate all existing tasks without submitting or retrying."""

    study = fq.twin.load_region_holdout_study(STUDY_PATH)
    submissions = tuple(
        fq.twin.load_submission(_submission_path(index))
        for index in range(1, study.planned_task_count + 1)
    )
    results = tuple(provider.fetch_result(item.receipt) for item in submissions)
    boundary = study.reference_suite.planned_task_count
    evaluation = study.validate_results(
        submissions[:boundary],
        results[:boundary],
        submissions[boundary:],
        results[boundary:],
        reference_circuits=reference_circuits(),
        holdout_circuits=holdout_circuits(),
        confidence_level=0.95,
    )
    fq.twin.dump_circuit_support(
        evaluation.reference_evaluation.to_circuit_support(),
        REFERENCE_SUPPORT_PATH,
    )
    fq.twin.dump_circuit_support(
        evaluation.holdout_evaluation.to_circuit_support(),
        HOLDOUT_SUPPORT_PATH,
    )

    print("tasks:", evaluation.task_count)
    print("shots:", evaluation.total_shots)
    print(
        "reference Twin-QPU agreement:",
        f"{evaluation.reference_twin_qpu_agreement:.2%}",
    )
    print(
        "holdout Twin-QPU agreement:",
        f"{evaluation.holdout_twin_qpu_agreement:.2%}",
    )
    print(
        "holdout Twin-QPU TV increase:",
        f"{evaluation.holdout_twin_qpu_tv_increase:.2%}",
    )
    print(
        "holdout ideal-SV-QPU agreement:",
        f"{evaluation.holdout_ideal_qpu_agreement:.2%}",
    )
    print(
        "holdout QPU repeatability:",
        f"{evaluation.holdout_qpu_repeatability:.2%}",
    )
    print(
        "holdout simultaneous TV error bound:",
        f"{evaluation.holdout_simultaneous_tv_error_bound:.2%}",
    )
    print("confidence level:", f"{evaluation.confidence_level:.2%}")
    print("reference support:", REFERENCE_SUPPORT_PATH)
    print("holdout support:", HOLDOUT_SUPPORT_PATH)


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
