"""Generate identity-bound Twin evidence from one Quafu hardware result.

Set ``QUAFU_API_TOKEN`` before running this example. The selected physical
qubits must exist in the current Shenglian calibration and remain connected.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import flagquantum as fq
from flagquantum.remote.qpu import DeploymentResult, ProviderTaskHandle, QuafuProvider

TARGET = "quafu:Shenglian"
BACKEND = "Shenglian"
QUBITS = (20, 27)
SHOTS = 1024
EVIDENCE_PATH = Path("twin-evidence.json")


def _wait_for_result(
    provider: QuafuProvider,
    receipt: ProviderTaskHandle,
    *,
    timeout: float = 600.0,
    poll_interval: float = 3.0,
) -> DeploymentResult:
    """Poll one submitted task without resubmitting it."""

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

    provider = QuafuProvider()
    chip_info = provider.fetch_chip_info(BACKEND)
    twin = fq.twin.from_quafu_chip_info(
        chip_info,
        target=TARGET,
        qubits=QUBITS,
    )
    circuit = fq.Circuit(2).h(0).cx(0, 1)

    experiment = fq.twin.TwinExperiment.prepare(
        twin,
        circuit,
        name="flagquantum-twin-bell",
        shots=SHOTS,
    )
    receipt = experiment.submit(provider)
    print("submitted task:", receipt.task_id, flush=True)
    result = _wait_for_result(provider, receipt)
    hardware_report = experiment.validate_result(result, receipt=receipt)
    evidence = experiment.evidence_from_report(
        hardware_report,
        circuit=circuit,
        confidence_level=0.95,
    )
    fq.twin.dump_evidence(evidence, EVIDENCE_PATH)

    restored = fq.twin.load_evidence(EVIDENCE_PATH)
    report = twin.evidence_report(circuit, evidence=restored)
    print("status:", report.status)
    print("TV error bound:", report.tv_error_bound)
    print("confidence level:", report.confidence_level)
    print("evidence:", EVIDENCE_PATH)


if __name__ == "__main__":
    main()
