"""Validate a Twin candidate across three fixed Shenglian circuits.

Set ``QUAFU_API_TOKEN`` and create ``twin-incumbent.json`` first. The workflow
uses separate commands so every accepted task is checkpointed before another
submission can begin:

``python examples/remote/quafu_twin_candidate_suite.py prepare``
``python examples/remote/quafu_twin_candidate_suite.py submit 1``
``python examples/remote/quafu_twin_candidate_suite.py submit 2``
``python examples/remote/quafu_twin_candidate_suite.py submit 3``
``python examples/remote/quafu_twin_candidate_suite.py evaluate``

Only ``submit`` creates hardware work, exactly one 1,024-shot task per call.
``evaluate`` restores and polls those three tasks without a submission path.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import flagquantum as fq
from flagquantum.remote.qpu import DeploymentResult, ProviderTaskHandle, QuafuProvider

TARGET = "quafu:Shenglian"
BACKEND = "Shenglian"
SHOTS = 1024
INCUMBENT_PATH = Path("twin-incumbent.json")
SUITE_PATH = Path("twin-candidate-suite.json")
SUBMISSION_PREFIX = "twin-candidate-suite-submission"


def _circuits() -> tuple[fq.Circuit, ...]:
    return (
        fq.Circuit(2).h(0).cx(0, 1),
        fq.Circuit(2).x(0).cx(0, 1),
        fq.Circuit(2).h(1).cx(1, 0).x(1),
    )


def _submission_path(index: int) -> Path:
    return Path(f"{SUBMISSION_PREFIX}-{index:02d}.json")


def _wait_for_result(
    provider: QuafuProvider,
    receipt: ProviderTaskHandle,
    *,
    timeout: float = 600.0,
    poll_interval: float = 3.0,
) -> DeploymentResult:
    deadline = time.monotonic() + timeout
    previous_status: str | None = None
    while True:
        status = provider.query_status(receipt)
        if status != previous_status:
            print(receipt.task_id, "status:", status, flush=True)
            previous_status = status
        if status in {"Finished", "Completed", "Done"}:
            return provider.fetch_result(receipt)
        if status in {"Failed", "Cancelled", "Canceled"}:
            raise RuntimeError(f"Quafu task ended with status {status!r}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Quafu task did not finish before the deadline")
        time.sleep(min(poll_interval, remaining))


def prepare(provider: QuafuProvider) -> None:
    """Freeze the models and all predictions without submitting a task."""

    incumbent = fq.twin.load_twin(INCUMBENT_PATH)
    if (
        incumbent.snapshot.provider != "quafu"
        or incumbent.snapshot.backend_name != BACKEND
    ):
        raise RuntimeError(f"The incumbent Twin must target {TARGET}")
    candidate = fq.twin.from_quafu_chip_info(
        provider.fetch_chip_info(BACKEND),
        target=TARGET,
        qubits=incumbent.snapshot.physical_qubits,
    )
    suite = fq.twin.prepare_candidate_suite(
        incumbent,
        candidate,
        _circuits(),
        name="flagquantum-twin-candidate-suite",
        shots=SHOTS,
    )
    fq.twin.dump_candidate_suite(suite, SUITE_PATH)
    print("saved offline candidate suite:", SUITE_PATH)
    print("circuit count:", suite.circuit_count)


def submit_one(provider: QuafuProvider, index: int) -> None:
    """Submit and immediately checkpoint exactly one selected suite trial."""

    suite = fq.twin.load_candidate_suite(SUITE_PATH)
    if not 1 <= index <= suite.circuit_count:
        raise ValueError(f"index must be in [1, {suite.circuit_count}]")
    destination = _submission_path(index)
    if destination.exists():
        raise RuntimeError(f"{destination} already exists; do not resubmit")
    trial = suite.trials[index - 1]
    receipt = trial.experiment.submit(provider)
    print("provider accepted task:", receipt.task_id, flush=True)
    submission = fq.twin.TwinCandidateSubmission.from_receipt(trial, receipt)
    fq.twin.dump_candidate_submission(submission, destination)
    print("saved resumable binding:", destination, flush=True)


def evaluate(provider: QuafuProvider) -> None:
    """Fetch and evaluate all checkpointed tasks without submitting work."""

    suite = fq.twin.load_candidate_suite(SUITE_PATH)
    submissions = tuple(
        fq.twin.load_candidate_submission(_submission_path(index))
        for index in range(1, suite.circuit_count + 1)
    )
    results = tuple(
        _wait_for_result(provider, submission.receipt) for submission in submissions
    )
    evaluation = suite.validate_results(
        submissions,
        results,
        circuits=_circuits(),
        confidence_level=0.95,
    )
    print("suite decision:", evaluation.decision)
    print("circuits:", evaluation.circuit_count)
    print("total shots:", evaluation.total_shots)
    print(
        "incumbent Twin-QPU agreement:",
        f"{evaluation.mean_incumbent_qpu_agreement:.2%}",
    )
    print(
        "candidate Twin-QPU agreement:",
        f"{evaluation.mean_candidate_qpu_agreement:.2%}",
    )
    print("ideal SV-QPU agreement:", f"{evaluation.mean_ideal_qpu_agreement:.2%}")
    print(
        "95% mean improvement interval:",
        f"[{evaluation.candidate_improvement_lower_bound:.2%}, "
        f"{evaluation.candidate_improvement_upper_bound:.2%}]",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "submit", "evaluate"))
    parser.add_argument("index", nargs="?", type=int)
    args = parser.parse_args()
    if not os.getenv("QUAFU_API_TOKEN"):
        raise RuntimeError("Set QUAFU_API_TOKEN before contacting Quafu SQC")
    provider = QuafuProvider()
    if args.command == "prepare":
        if args.index is not None:
            parser.error("prepare does not accept an index")
        prepare(provider)
    elif args.command == "submit":
        if args.index is None:
            parser.error("submit requires one circuit index")
        submit_one(provider, args.index)
    else:
        if args.index is not None:
            parser.error("evaluate does not accept an index")
        evaluate(provider)


if __name__ == "__main__":
    main()
