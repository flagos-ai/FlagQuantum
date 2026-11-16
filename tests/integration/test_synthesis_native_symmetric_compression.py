"""Validate compression's complete action and its retained source preparation."""

from pathlib import Path

import pytest

from examples.synthesis.community.paper_implementation_project.quantum_compression_algorithm_for_symmetric_states.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.community.paper_implementation_project.quantum_compression_algorithm_for_symmetric_states.main import (
    build_program,
    compress,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, X, allocate, qfunc

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/community/paper_implementation_project/quantum_compression_algorithm_for_symmetric_states/quantum_compression_algorithm_for_symmetric_states.qmod"
)


@qfunc
def probe(bits: list[float], q: Output[QArray[QBit]]) -> None:
    allocate(3, q)
    for bit in range(3):
        if bits[bit] != 0:
            X(q[bit])
    compress(q)


def test_untouched_original_and_native_match_complex_state() -> None:
    reference = reference_circuit()
    original, native = synth.synthesize(SOURCE), build_program()
    for program in (original, native):
        compare_states(program, reference, ("x",))
        assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("basis", range(8))
def test_native_compression_matches_every_independent_basis_column(basis: int) -> None:
    native = synth.synthesize(
        probe,
        classical_inputs={"bits": [float((basis >> bit) & 1) for bit in range(3)]},
        max_workspace=0,
    )
    compare_states(native, reference_circuit(basis), ("q",))
