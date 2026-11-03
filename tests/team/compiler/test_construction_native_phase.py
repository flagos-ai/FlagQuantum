"""Native predicate phase through the verified register call graph."""

import math

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._construction.frontend_program import capture_program
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.synth import (
    Const,
    Output,
    QBit,
    QNum,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)

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
    phase(x, math.inf)


@qfunc
def uninitialized_phase(x: Output[QBit]) -> None:
    phase(x, 0.8)
    allocate(x)


@pytest.mark.parametrize(
    "definition,match",
    [
        (invalid_coefficient, "finite static real"),
        (uninitialized_phase, "uninitialized"),
    ],
)
def test_rejects_invalid_phase_operands(definition, match) -> None:
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


@qfunc
def weighted_cost(x: Const[QNum[2]], y: Const[QNum[2]]) -> None:
    phase(x * x + y, 0.8)


@qfunc
def weighted_main(x: Output[QNum[2]], y: Output[QNum[2]]) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    weighted_cost(x, y)
    phase(y, -0.3)
    phase(-0.5, 1.9)


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_weighted_polynomial_and_constant_global_phase(objective) -> None:
    program = synth.synthesize(weighted_main, objective=objective, max_qubits=24)
    state = fq.run(program).statevector().reshape(-1)
    expected = torch.zeros_like(state)
    scratch = program.circuit.n_wires - 4
    for x in range(4):
        for y in range(4):
            angle = 0.8 * (x * x + y) - 0.3 * y - 0.95
            expected[(4 * x + y) << scratch] = (
                complex(math.cos(angle), math.sin(angle)) / 4
            )
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


@qfunc
def rz_equivalent(q: Output[QBit]) -> None:
    allocate(q)
    hadamard_transform(q)
    phase(q, 1.9)
    phase(-0.5, 1.9)


def test_weighted_and_constant_phase_matches_rz_without_rephasing() -> None:
    program = synth.synthesize(rz_equivalent)
    qmod = synth.synthesize(
        "qfunc main(output q:qbit) {allocate(q);H(q);RZ(1.9,q);}",
        source_format="qmod",
    )
    reference = fq.Circuit(1, dtype=torch.complex128).h(0).rz(0, theta=1.9)
    actual = fq.run(program).statevector().reshape(-1)
    expected = torch.zeros_like(actual)
    stride = 1 << (program.circuit.n_wires - 1)
    expected[::stride] = fq.run(reference).statevector().reshape(-1)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        fq.run(qmod).statevector().reshape(-1), expected[::stride], atol=1e-12, rtol=0
    )


def test_weighted_phase_reports_actual_workspace_and_target_limits() -> None:
    program = synth.synthesize(weighted_main, max_qubits=24)
    workspace = program.report["workspace_qubits"]
    assert isinstance(workspace, int) and workspace > 0
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(weighted_main, max_workspace=workspace - 1)
    with pytest.raises(ValueError, match="phase output admits only"):
        synth.synthesize(weighted_main, basis=("x", "cx", "phase"))
