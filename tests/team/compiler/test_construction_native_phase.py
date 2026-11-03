"""Native predicate phase through the verified register call graph."""

import math

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._construction.frontend_program import capture_program
from flagquantum.compiler._construction.program_types import phase
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.synth import Const, Output, QBit, allocate, hadamard_transform, qfunc

pytestmark = pytest.mark.unit


@qfunc
def mark(x: Const[QBit], y: Const[QBit]) -> None:
    phase(x * y, math.pi)


@qfunc
def nested(x: Const[QBit], y: Const[QBit]) -> None:
    mark(x, y)
    phase(y, math.pi)


@qfunc
def main(x: Output[QBit], y: Output[QBit]) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    nested(x, y)
    mark(x, y)
    mark(x, y)


@qfunc
def invalid_coefficient(x: Output[QBit]) -> None:
    allocate(x)
    phase(x, 0.8)


@qfunc
def invalid_width(x: Output[QBit]) -> None:
    allocate(x)
    phase(x + x, math.pi)


@pytest.mark.parametrize(
    "definition,match",
    [(invalid_coefficient, "static pi"), (invalid_width, "one-bit return")],
)
def test_rejects_unimplemented_phase_profiles(definition, match) -> None:
    with pytest.raises(HybridVerificationError, match=match):
        capture_program(definition)


def test_nested_and_repeated_native_phase_matches_qmod_and_exact_state() -> None:
    source = """qfunc mark(const x:qbit,const y:qbit) { phase(x*y,pi); }
    qfunc nested(const x:qbit,const y:qbit) { mark(x,y); phase(y,pi); }
    qfunc main(output x:qbit,output y:qbit) {
        allocate(x); allocate(y); H(x); H(y); nested(x,y); mark(x,y); mark(x,y);
    }"""
    for program in (
        synth.synthesize(main),
        synth.synthesize(source, source_format="qmod"),
    ):
        state = fq.run(program).statevector().reshape(-1)
        expected = torch.zeros_like(state)
        scratch = program.circuit.n_wires - 2
        for x in range(2):
            for y in range(2):
                expected[(2 * x + y) << scratch] = (-1) ** (x * y + y) / 2
        torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
