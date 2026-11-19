"""Reuse independent Fourier-parity and Fourier-coefficient Circuit references."""

from math import sqrt

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.qct_qst.overview.main import (
    original_data,
    run_workflow,
)
from examples.synthesis.functions.qct_qst.reference import transform_circuit


def reference_circuits() -> dict[str, fq.Circuit]:
    cosine, sine, data = original_data()
    combined = cosine.tolist() + sine.tolist()
    combined = [value / sqrt(2) for value in combined]
    return {
        "type_one": transform_circuit(combined, "qct_qst_type1"),
        "type_two": transform_circuit(data.tolist(), "qct_qst_type2"),
        "cosine": transform_circuit(data.tolist(), "qct_type2"),
        "sine": transform_circuit(data.tolist(), "qst_type2"),
    }


def run_example() -> None:
    circuits = reference_circuits()
    for name, program in run_workflow().items():
        compare_states(
            program, circuits[name], ("x", "q") if name == "type_two" else ("x",)
        )


if __name__ == "__main__":
    run_example()
