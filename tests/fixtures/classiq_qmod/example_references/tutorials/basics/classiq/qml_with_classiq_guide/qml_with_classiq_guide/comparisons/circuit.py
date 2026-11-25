"""Compare the untouched Qmod and native qfunc against one independent RX."""

from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.tutorials.basics.qml.main import (
    main,
)
from flagquantum.synth import synthesize
from tools.validate_native_source_states import compare_states


def run_example() -> None:
    for input_0, weight_0 in ((0.37, -0.81), (1.2, 6.1)):
        bindings = {"input_0": input_0, "weight_0": weight_0}
        native = synthesize(main, classical_inputs=bindings)
        reference = fq.Circuit(1, dtype=torch.complex128).rx(
            0, theta=input_0 + weight_0
        )
        compare_states(
            native,
            Path(__file__).with_name("source.qmod"),
            reference,
            ("res",),
            classical_inputs=bindings,
        )
        assert native.report["workspace_qubits"] == 0


if __name__ == "__main__":
    run_example()
