"""Independent Fourier-based Circuit comparison for qct_qst_type1."""

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.qct_qst.qct_qst_type1.main import (
    AMPLITUDES,
    build_program,
)
from examples.synthesis.functions.qct_qst.reference import transform_circuit


def reference_circuit() -> fq.Circuit:
    return transform_circuit(AMPLITUDES, "qct_qst_type1")


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("x",))


if __name__ == "__main__":
    run_example()
