"""Build and inspect an IQM dynamic task before explicitly submitting it."""

from __future__ import annotations

import json
import os

import flagquantum as fq


def main() -> None:
    device_arn = os.environ["FLAGQUANTUM_BRAKET_DEVICE_ARN"]
    dynamic_qubit_groups = tuple(
        tuple(int(wire) for wire in group)
        for group in json.loads(os.environ["FLAGQUANTUM_IQM_DYNAMIC_GROUPS"])
    )
    provider = fq.deployment.AmazonBraketProvider(
        device_arn,
        dynamic_qubit_groups=dynamic_qubit_groups,
    )
    circuit = fq.experimental.dynamic.DynamicCircuit(2)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("x", 1, classical_bit=0)
    package = fq.experimental.dynamic.create_dynamic_deployment_package(
        circuit,
        backend=provider.backend,
        name="iqm_feedback",
        shots=100,
    )

    preview = provider.dry_run(package)
    print(preview.summary())
    print(preview.program)

    # Hardware submission can incur charges and therefore remains explicit.
    if os.environ.get("FLAGQUANTUM_SUBMIT_BRAKET") == "1":
        result = provider.run(package)
        print(result.counts)


if __name__ == "__main__":
    main()
