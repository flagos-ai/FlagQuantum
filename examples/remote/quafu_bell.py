"""Compile and run a Bell circuit on Quafu SQC.

Set ``QUAFU_API_TOKEN`` and install ``flagquantum-compiler-qsteed`` before
running this example. The selected chip name must exist in the current Quafu
account.
"""

from __future__ import annotations

import os

import flagquantum as fq


def main() -> None:
    if not os.getenv("QUAFU_API_TOKEN"):
        raise RuntimeError("Set QUAFU_API_TOKEN before submitting to Quafu SQC")

    circuit = fq.Circuit(2).h(0).cx(0, 1)
    result = fq.run(
        circuit,
        compiler="qsteed",
        target="quafu:Baihua",
        shots=1024,
        name="flagquantum bell",
    )

    print("task:", result.provenance["task_id"])
    print("counts:", result.counts[0])


if __name__ == "__main__":
    main()
