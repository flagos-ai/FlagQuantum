"""Borrowed Fourier callbacks preserve inverse and relative scalar phase."""

import cmath

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.qft.comparisons.circuit import fourier_matrix
from flagquantum import synth
from flagquantum.compiler._construction.frontend_python import _QuantumFunction
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration


def definition(width: int, inverse: int, phase: float) -> _QuantumFunction:
    return program_definition(
        "from flagquantum.synth import *\n"
        "@qfunc\ndef transform(q:QArray[QBit], reverse:int):\n"
        "    if reverse == 1:\n"
        "        with adjoint():\n            phase(1,0.23)\n            qft(q)\n"
        "    else:\n        phase(1,0.23)\n        qft(q)\n"
        "@qfunc\ndef main(c:Output[QBit],q:Output[QArray[QBit]]):\n"
        f"    allocate(c)\n    allocate({width},q)\n"
        f"    H(c)\n    PHASE({phase},c)\n    hadamard_transform(q)\n"
        f"    for i in range({width}):\n        PHASE(0.29*(i+1),q[i])\n"
        f"    control(c,lambda:transform(q,{inverse}))\n    H(c)\n"
    )


@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("inverse", [0, 1])
@pytest.mark.parametrize("phase", [0.31, -0.67])
def test_borrowed_fourier_complex_state_and_control_interference(
    width: int, inverse: int, phase: float
) -> None:
    program = synth.synthesize(definition(width, inverse, phase), max_workspace=0)
    reference = fq.Circuit(width + 1, dtype=torch.complex128).h(0).phase(0, theta=phase)
    for i in range(width):
        reference.h(width - i).phase(width - i, theta=0.29 * (i + 1))
    transform = fourier_matrix(width)
    if inverse:
        transform = transform.conj().T
    transform = transform * cmath.exp((-1 if inverse else 1) * 0.23j)
    size = 1 << width
    controlled = torch.zeros((2 * size, 2 * size), dtype=torch.complex128)
    controlled[:size, :size] = torch.eye(size, dtype=torch.complex128)
    controlled[size:, size:] = transform
    reference.any(*range(width + 1), matrix=controlled).h(0)
    compare_states(program, reference, ("c", "q"))
    assert program.report["workspace_qubits"] == 0


def test_borrowed_fourier_resources_are_not_hidden_by_callback() -> None:
    with pytest.raises(
        (HybridVerificationError, PlanningError), match="budget|limit|infeasible"
    ):
        synth.synthesize(definition(4, 1, 0.31), max_operations=2)
    with pytest.raises(PlanningError, match="memory|budget"):
        synth.synthesize(definition(4, 1, 0.31), memory_limit_bytes=16)


def test_borrowed_fourier_control_cannot_alias_target() -> None:
    source = (
        "from flagquantum.synth import *\n"
        "@qfunc\ndef transform(q:QArray[QBit]):\n"
        "    with adjoint():\n        qft(q)\n"
        "@qfunc\ndef main(q:Output[QArray[QBit]]):\n"
        "    allocate(4,q)\n    control(q[0],lambda:transform(q))\n"
    )
    with pytest.raises(HybridVerificationError, match="disjoint"):
        synth.synthesize(program_definition(source))
