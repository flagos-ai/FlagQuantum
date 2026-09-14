"""Compare two frozen Twins using one future Quafu hardware result.

Set ``QUAFU_API_TOKEN`` and create ``twin-incumbent.json`` with
``fq.twin.dump_twin`` before running this example. Running the example
explicitly submits one 1,024-shot Shenglian task.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import flagquantum as fq
from flagquantum.remote.qpu import DeploymentResult, ProviderTaskHandle, QuafuProvider

TARGET = "quafu:Shenglian"
BACKEND = "Shenglian"
SHOTS = 1024
INCUMBENT_PATH = Path("twin-incumbent.json")


def _wait_for_result(
    provider: QuafuProvider,
    receipt: ProviderTaskHandle,
    *,
    timeout: float = 600.0,
    poll_interval: float = 3.0,
) -> DeploymentResult:
    """Poll the one submitted task without resubmitting it."""

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


def main() -> None:
    if not os.getenv("QUAFU_API_TOKEN"):
        raise RuntimeError("Set QUAFU_API_TOKEN before submitting to Quafu SQC")

    incumbent = fq.twin.load_twin(INCUMBENT_PATH)
    if (
        incumbent.snapshot.provider != "quafu"
        or incumbent.snapshot.backend_name != BACKEND
    ):
        raise RuntimeError(f"The incumbent Twin must target {TARGET}")

    provider = QuafuProvider()
    candidate = fq.twin.from_quafu_chip_info(
        provider.fetch_chip_info(BACKEND),
        target=TARGET,
        qubits=incumbent.snapshot.physical_qubits,
    )
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    trial = fq.twin.prepare_candidate_trial(
        incumbent,
        candidate,
        circuit,
        name="flagquantum-twin-candidate-bell",
        shots=SHOTS,
    )

    receipt = trial.experiment.submit(provider)
    print("submitted one comparison task:", receipt.task_id, flush=True)
    result = _wait_for_result(provider, receipt)
    evaluation = trial.validate_result(
        result,
        receipt=receipt,
        circuit=circuit,
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
    print("candidate improvement:", f"{evaluation.candidate_improvement:.2%}")
    print(
        "95% candidate-improvement interval:",
        f"[{evaluation.candidate_improvement_lower_bound:.2%}, "
        f"{evaluation.candidate_improvement_upper_bound:.2%}]",
    )


if __name__ == "__main__":
    main()
