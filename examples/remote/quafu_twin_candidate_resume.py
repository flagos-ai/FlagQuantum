"""Submit once, then resume a frozen Twin candidate comparison.

Set ``QUAFU_API_TOKEN`` and create ``twin-incumbent.json`` with
``fq.twin.dump_twin`` before using this example.

``submit`` explicitly creates one 1,024-shot Shenglian task and immediately
writes a private checkpoint. ``resume`` only loads that checkpoint, polls the
same task, and validates its result; it never submits hardware work.
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
SUBMISSION_PATH = Path("twin-candidate-submission.json")


def _circuit() -> fq.Circuit:
    return fq.Circuit(2).h(0).cx(0, 1)


def _wait_for_result(
    provider: QuafuProvider,
    receipt: ProviderTaskHandle,
    *,
    timeout: float = 600.0,
    poll_interval: float = 3.0,
) -> DeploymentResult:
    """Poll one restored receipt without resubmitting it."""

    deadline = time.monotonic() + timeout
    previous_status: str | None = None
    while True:
        status = provider.query_status(receipt)
        if status != previous_status:
            print("task status:", status, flush=True)
            previous_status = status
        if status in {"Finished", "Completed", "Done"}:
            return provider.fetch_result(receipt)
        if status in {"Failed", "Cancelled", "Canceled"}:
            raise RuntimeError(f"Quafu task ended with status {status!r}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Quafu task did not finish before the deadline")
        time.sleep(min(poll_interval, remaining))


def submit_once(provider: QuafuProvider) -> None:
    """Create exactly one task and persist its complete candidate binding."""

    if SUBMISSION_PATH.exists():
        raise RuntimeError(f"{SUBMISSION_PATH} already exists; use the resume command")
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
    trial = fq.twin.prepare_candidate_trial(
        incumbent,
        candidate,
        _circuit(),
        name="flagquantum-twin-candidate-bell",
        shots=SHOTS,
    )

    receipt = trial.experiment.submit(provider)
    print("submitted one comparison task:", receipt.task_id, flush=True)
    submission = fq.twin.TwinCandidateSubmission.from_receipt(trial, receipt)
    fq.twin.dump_candidate_submission(submission, SUBMISSION_PATH)
    print("saved resumable binding:", SUBMISSION_PATH, flush=True)


def resume(provider: QuafuProvider) -> None:
    """Read and validate the existing task without a submission path."""

    submission = fq.twin.load_candidate_submission(SUBMISSION_PATH)
    result = _wait_for_result(provider, submission.receipt)
    evaluation = submission.validate_result(
        result,
        circuit=_circuit(),
        confidence_level=0.95,
    )
    print("decision:", evaluation.decision)
    print(
        "incumbent Twin-QPU agreement:",
        f"{1 - evaluation.incumbent_hardware_total_variation:.2%}",
    )
    print(
        "candidate Twin-QPU agreement:",
        f"{1 - evaluation.candidate_hardware_total_variation:.2%}",
    )
    print(
        "95% candidate-improvement interval:",
        f"[{evaluation.candidate_improvement_lower_bound:.2%}, "
        f"{evaluation.candidate_improvement_upper_bound:.2%}]",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("submit", "resume"))
    args = parser.parse_args()
    if not os.getenv("QUAFU_API_TOKEN"):
        raise RuntimeError("Set QUAFU_API_TOKEN before contacting Quafu SQC")

    provider = QuafuProvider()
    if args.command == "submit":
        submit_once(provider)
    else:
        resume(provider)


if __name__ == "__main__":
    main()
