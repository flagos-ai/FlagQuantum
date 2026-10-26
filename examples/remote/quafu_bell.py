"""Submit a Bell circuit through the maintained Quafu SDK.

Set ``QUAFU_API_KEY`` before running this example. The selected chip name must
exist in the current Quafu account.
"""

from __future__ import annotations

import os

import flagquantum as fq


def main() -> None:
    if not os.getenv("QUAFU_API_KEY"):
        raise RuntimeError("Set QUAFU_API_KEY before submitting through quafu-sdk")

    circuit = fq.Circuit(2).h(0).cx(0, 1)
    result = fq.run(
        circuit,
        target="quafu:Baihua",
        shots=1024,
        name="flagquantum bell",
    )

    print("task:", result.provenance["task_id"])
    print("service compiler:", result.provenance["service_compiler"])
    print("counts:", result.counts[0])


if __name__ == "__main__":
    main()
