"""Compile and submit a Bell circuit to Quafu SQC.

Set ``QUAFU_API_TOKEN`` before running this example.  Quafu tokens must never be
committed to source control and currently expire after 30 days.
Install ``flagquantum-compiler-qsteed`` before running this example.
"""

from __future__ import annotations

import os

import flagquantum as fq
import flagquantum.deployment as fqd
from flagquantum.remote import QuafuProvider


def main() -> None:
    if not os.getenv("QUAFU_API_TOKEN"):
        raise RuntimeError("Set QUAFU_API_TOKEN before submitting to Quafu SQC")

    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)
    compiled = fq.compile(
        circuit,
        compiler="qsteed",
        target="quafu:ScQ-P10",
    )
    provider = QuafuProvider(result_timeout=1800)
    result = fqd.deploy_circuit(
        compiled,
        provider,
        shots=1024,
    )
    print(result.handle.task_id, result.counts)


if __name__ == "__main__":
    main()
