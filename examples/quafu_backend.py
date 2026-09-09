"""Submit a Bell circuit to Quafu SQC with the FlagQuantum deployment API.

Set ``QUAFU_API_TOKEN`` before running this example.  Quafu tokens must never be
committed to source control and currently expire after 30 days.
"""

from __future__ import annotations

import os

import flagquantum as fq
import flagquantum.deployment as fqd
from flagquantum.remote import QuafuProvider


def main() -> None:
    if not os.getenv("QUAFU_API_TOKEN"):
        raise RuntimeError("Set QUAFU_API_TOKEN before submitting to Quafu SQC")

    provider = QuafuProvider(result_timeout=1800)
    available = provider.discover_backends(2)
    online = [
        backend
        for backend in available
        if str(backend.metadata["queue_status"]).lower()
        not in {"offline", "maintenance"}
    ]
    if not online:
        raise RuntimeError("No suitable Quafu backend is currently online")
    backend = min(
        online,
        key=lambda item: (
            item.metadata["queue_status"]
            if isinstance(item.metadata["queue_status"], int)
            else float("inf")
        ),
    )

    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)
    package = fqd.create_deployment_package(
        circuit,
        backend=backend,
        shots=1024,
        metadata={
            "provider_compile": True,
            "provider_options": {
                "compiler": "quarkcircuit",
                "correct": False,
                "open_dd": None,
                "target_qubits": [],
            },
        },
    )
    result = provider.run(package)
    print(result.handle.task_id, result.counts)


if __name__ == "__main__":
    main()
