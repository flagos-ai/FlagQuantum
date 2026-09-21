"""Prospectively validate one connected regional Twin circuit on Quafu.

Set ``QUAFU_API_TOKEN`` before running. Adjust the cell artifact paths, physical
mapping, and backend to artifacts captured from the same calibration. Running
this example explicitly submits exactly ``REPETITIONS`` hardware tasks.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import flagquantum as fq
from flagquantum.remote.qpu import DeploymentResult, ProviderTaskHandle, QuafuProvider
from flagquantum.twin import TwinHardwareReport

BACKEND = "Shenglian"
PHYSICAL_QUBITS = (20, 27, 34)
SHOTS = 1024
REPETITIONS = 2
CELL_A_TWIN = Path("cell-a-twin.json")
CELL_A_SUPPORT = Path("cell-a-support.json")
CELL_B_TWIN = Path("cell-b-twin.json")
CELL_B_SUPPORT = Path("cell-b-support.json")
VALIDATION_PATH = Path("regional-twin-validation.json")
SUPPORT_PATH = Path("regional-twin-support.json")


def _wait_for_result(
    provider: QuafuProvider,
    receipt: ProviderTaskHandle,
    *,
    timeout: float = 600.0,
    poll_interval: float = 3.0,
) -> DeploymentResult:
    """Poll one submitted task without resubmitting it."""

    deadline = time.monotonic() + timeout
    while True:
        status = provider.query_status(receipt)
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

    twin_a = fq.twin.load_twin(CELL_A_TWIN)
    support_a = fq.twin.load_circuit_support(CELL_A_SUPPORT)
    twin_b = fq.twin.load_twin(CELL_B_TWIN)
    support_b = fq.twin.load_circuit_support(CELL_B_SUPPORT)
    region_twin = fq.twin.compose_region_twin(
        ((twin_a, support_a), (twin_b, support_b))
    )
    if region_twin.twin.snapshot.backend_name != BACKEND:
        raise RuntimeError("regional Twin artifacts identify a different backend")

    circuit = fq.Circuit(3).h(0).cx(0, 1).cx(1, 2)
    experiment = region_twin.prepare_experiment(
        circuit,
        physical_qubits=PHYSICAL_QUBITS,
        name="flagquantum-regional-ghz",
        shots=SHOTS,
    )
    provider = QuafuProvider()
    reports: list[TwinHardwareReport] = []
    for repetition in range(1, REPETITIONS + 1):
        receipt = experiment.submit(provider)
        print(
            f"submitted repetition {repetition}/{REPETITIONS}:",
            receipt.task_id,
            flush=True,
        )
        result = _wait_for_result(provider, receipt)
        reports.append(experiment.validate_result(result, receipt=receipt))

    series = experiment.validation_series(reports, circuit=circuit)
    support = region_twin.support_from_validation_series(
        series,
        circuit,
        physical_qubits=PHYSICAL_QUBITS,
    )
    fq.twin.dump_validation_series(series, VALIDATION_PATH)
    fq.twin.dump_circuit_support(support, SUPPORT_PATH)

    report = support.evidence_report(region_twin.twin, circuit)
    print("status:", report.status)
    print("Twin-QPU agreement:", f"{series.mean_twin_qpu_agreement:.2%}")
    print("ideal-SV-QPU agreement:", f"{series.mean_ideal_qpu_agreement:.2%}")
    print("QPU repeatability:", f"{series.mean_qpu_repeatability:.2%}")
    print(
        "simultaneous shot radius:", f"{series.simultaneous_finite_shot_tv_radius:.2%}"
    )
    print("verified TV error bound:", f"{series.verified_tv_error_bound:.2%}")
    print("confidence level:", f"{series.confidence_level:.2%}")


if __name__ == "__main__":
    main()
