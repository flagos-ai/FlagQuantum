"""Generate identity-bound Twin evidence from repeated Quafu hardware results.

Set ``QUAFU_API_TOKEN`` before running this example. The selected physical
qubits must exist in the current Shenglian calibration and remain connected.
Running the example explicitly submits ``REPETITIONS`` hardware tasks.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import flagquantum as fq
from flagquantum.remote.qpu import DeploymentResult, ProviderTaskHandle, QuafuProvider
from flagquantum.twin import TwinHardwareReport

TARGET = "quafu:Shenglian"
BACKEND = "Shenglian"
QUBITS = (20, 27)
SHOTS = 1024
REPETITIONS = 2
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
    hardware_reports: list[TwinHardwareReport] = []
    for repetition in range(1, REPETITIONS + 1):
        receipt = experiment.submit(provider)
        print(
            f"submitted repetition {repetition}/{REPETITIONS}:",
            receipt.task_id,
            flush=True,
        )
        result = _wait_for_result(provider, receipt)
        hardware_reports.append(experiment.validate_result(result, receipt=receipt))

    series = experiment.validation_series(
        hardware_reports,
        circuit=circuit,
        confidence_level=0.95,
    )
    evidence = series.to_evidence()
    fq.twin.dump_evidence(evidence, EVIDENCE_PATH)

    restored = fq.twin.load_evidence(EVIDENCE_PATH)
    report = twin.evidence_report(circuit, evidence=restored)
    print("status:", report.status)
    print("Twin-QPU agreement:", f"{series.mean_twin_qpu_agreement:.2%}")
    print("ideal-QPU agreement:", f"{series.mean_ideal_qpu_agreement:.2%}")
    repeatability = series.mean_qpu_repeatability
    print(
        "QPU repeatability:",
        "not available" if repeatability is None else f"{repeatability:.2%}",
    )
    print(
        "simultaneous shot radius:",
        f"{series.simultaneous_finite_shot_tv_radius:.2%}",
    )
    print("TV error bound:", report.tv_error_bound)
    print("confidence level:", report.confidence_level)
    print("evidence:", EVIDENCE_PATH)


if __name__ == "__main__":
    main()
