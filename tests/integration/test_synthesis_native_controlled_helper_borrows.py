"""Exact scalar/target borrowing, independent complex operators."""

import cmath
from dataclasses import replace
from typing import Annotated

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler.frontends._structured.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from flagquantum.synth import (
    RZ,
    Const,
    H,
    Output,
    QBit,
    QNum,
    allocate,
    control,
    phase,
    qfunc,
)

pytestmark = pytest.mark.integration


@qfunc
def payload(angle: float, gamma: float, target: QNum) -> None:
    H(target[0])
    RZ(angle, target[0])
    phase(1, gamma)


@qfunc
def nested_payload(angle: float, gamma: float, target: QNum) -> None:
    payload(angle, gamma, target)


@qfunc
def main_zero(
    ci: int, ti: int, angle: float, gamma: float, q: Output[Annotated[QNum, 4]]
) -> None:
    allocate(q)
    control(q[ci] == 0, lambda: nested_payload(angle, gamma, q[ti : ti + 1]))


@qfunc
def main_one(
    ci: int, ti: int, angle: float, gamma: float, q: Output[Annotated[QNum, 4]]
) -> None:
    allocate(q)
    control(q[ci] == 1, lambda: nested_payload(angle, gamma, q[ti : ti + 1]))


def independent_body(
    ci: int, ti: int, selected: int, angle: float, gamma: float
) -> fq.Circuit:
    h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) * 2**-0.5
    rz = torch.diag(
        torch.tensor(
            [cmath.exp(-0.5j * angle), cmath.exp(0.5j * angle)], dtype=torch.complex128
        )
    )
    u = cmath.exp(1j * gamma) * (rz @ h)
    identity = torch.eye(2, dtype=torch.complex128)
    matrix = torch.zeros((4, 4), dtype=torch.complex128)
    matrix[:2, :2] = u if selected == 0 else identity
    matrix[2:, 2:] = identity if selected == 0 else u
    return fq.Circuit(4, dtype=torch.complex128).any(3 - ci, 3 - ti, matrix=matrix)


@pytest.mark.parametrize(
    "ci,ti", [(c, t) for c in range(4) for t in range(4) if c != t]
)
@pytest.mark.parametrize("selected", (0, 1))
def test_all_basis_and_complex_coherence(ci: int, ti: int, selected: int) -> None:
    angle = 0.37 + 0.19 * (ci + ti)
    gamma = -0.23 - 0.11 * ci
    program = synth.synthesize(
        main_zero if selected == 0 else main_one,
        classical_inputs={"ci": ci, "ti": ti, "angle": angle, "gamma": gamma},
        max_qubits=5,
        max_workspace=1,
        memory_limit_bytes=64 * 1024**2,
    )
    body = independent_body(ci, ti, selected, angle, gamma).to_ir()
    logical = program.registers["q"]
    hidden = tuple(w for w in range(program.circuit.n_wires) if w not in logical)
    for word in (*range(16), None):
        prefix = fq.Circuit(4, dtype=torch.complex128)
        if word is None:
            for w in range(4):
                prefix.h(w).rz(w, 0.217 * (w + 1))
            prefix.cx(0, 3)
        else:
            for w in range(4):
                if word & (1 << (3 - w)):
                    prefix.x(w)
        before = prefix.to_ir().instructions
        expected = (
            fq.run(replace(body, instructions=before + body.instructions))
            .statevector()
            .flatten()
        )
        mapped = tuple(
            replace(op, wires=tuple(logical[w] for w in op.wires)) for op in before
        )
        actual = (
            fq.run(
                replace(
                    program.circuit, instructions=mapped + program.circuit.instructions
                )
            )
            .statevector()
            .reshape((2,) * program.circuit.n_wires)
            .permute((*logical, *hidden))
            .reshape(16, -1)
        )
        torch.testing.assert_close(actual[:, 0], expected, atol=1e-11, rtol=0)
        assert float(actual[:, 1:].abs().square().sum()) < 1e-22


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("selected", (0, 1))
def test_actual_selector_overlap_refuses(index: int, selected: int) -> None:
    with pytest.raises(HybridVerificationError, match="disjoint"):
        synth.synthesize(
            main_zero if selected == 0 else main_one,
            classical_inputs={"ci": index, "ti": index, "angle": 0.37, "gamma": -0.23},
        )


@pytest.mark.parametrize("budget", ("width", "operations", "depth", "memory"))
def test_resource_budgets_refuse(budget: str) -> None:
    inputs = {"ci": 0, "ti": 1, "angle": 0.37, "gamma": -0.23}
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if budget == "width":
            synth.synthesize(main_zero, classical_inputs=inputs, max_qubits=3)
        elif budget == "operations":
            synth.synthesize(main_zero, classical_inputs=inputs, max_operations=1)
        elif budget == "depth":
            synth.synthesize(main_zero, classical_inputs=inputs, max_depth=1)
        else:
            synth.synthesize(main_zero, classical_inputs=inputs, memory_limit_bytes=16)


@qfunc
def two_targets(first: QNum, second: QNum) -> None:
    H(first[0])
    RZ(0.37, second[0])


@qfunc
def duplicate_targets(q: Output[Annotated[QNum, 4]]) -> None:
    allocate(q)
    control(q[0] == 0, lambda: two_targets(q[1:2], q[1:2]))


@qfunc
def create_output(target: Output[QBit]) -> None:
    allocate(target)
    H(target)


@qfunc
def escaping_output(q: Output[Annotated[QNum, 4]], leaked: Output[QBit]) -> None:
    allocate(q)
    control(q[0] == 0, lambda: create_output(leaked))


def test_repeated_writable_aliases_still_refuse() -> None:
    with pytest.raises(HybridVerificationError, match="disjoint|alias"):
        synth.synthesize(duplicate_targets)


def test_output_escape_still_refuses() -> None:
    with pytest.raises(HybridVerificationError, match="control|borrow|output|Output"):
        synth.synthesize(escaping_output)


@qfunc
def composed_payload(target: QNum) -> None:
    H(target[0])
    RZ(0.43, target[3])
    synth.CX(target[1], target[2])
    phase(1, -0.37)


@qfunc
def composed_program(
    q: Output[Annotated[QNum, 3]], data: Output[Annotated[QNum, 2]]
) -> None:
    allocate(q)
    allocate(data)
    control(q[0] == 0, lambda: composed_payload([data, q[1:3]]))


def test_composed_view_preserves_all_32_basis_inputs_and_complex_coherence() -> None:
    program = synth.synthesize(
        composed_program, max_qubits=6, max_workspace=1, memory_limit_bytes=64 * 1024**2
    )
    # Target integer order is data[0], data[1], q[1], q[2]. Build the
    # controlled operator independently, retaining its branch scalar phase.
    target = torch.zeros((16, 16), dtype=torch.complex128)
    for word in range(16):
        for bit in range(2):
            after_h = (word & ~1) | bit
            after_cx = after_h ^ (4 if after_h & 2 else 0)
            amplitude: complex = (-1 if word & 1 and bit else 1) * 2**-0.5
            amplitude *= cmath.exp(0.5j * 0.43 * (1 if word & 8 else -1) - 0.37j)
            target[after_cx, word] = torch.tensor(amplitude, dtype=torch.complex128)
    controlled = torch.zeros((32, 32), dtype=torch.complex128)
    controlled[:16, :16] = target
    controlled[16:, 16:] = torch.eye(16, dtype=torch.complex128)
    reference = fq.Circuit(5, dtype=torch.complex128).any(
        2, 0, 1, 3, 4, matrix=controlled
    )
    body = reference.to_ir()
    logical = tuple(w for name in ("q", "data") for w in program.registers[name])
    hidden = tuple(w for w in range(program.circuit.n_wires) if w not in logical)
    for word in (*range(32), None):
        prefix = fq.Circuit(5, dtype=torch.complex128)
        for wire in range(5):
            if word is None:
                prefix.h(wire).rz(wire, 0.217 * (wire + 1))
            elif word & (1 << (4 - wire)):
                prefix.x(wire)
        if word is None:
            prefix.cx(0, 4)
        before = prefix.to_ir().instructions
        expected = (
            fq.run(replace(body, instructions=before + body.instructions))
            .statevector()
            .flatten()
        )
        mapped = tuple(
            replace(op, wires=tuple(logical[w] for w in op.wires)) for op in before
        )
        actual = (
            fq.run(
                replace(
                    program.circuit, instructions=mapped + program.circuit.instructions
                )
            )
            .statevector()
            .reshape((2,) * program.circuit.n_wires)
            .permute((*logical, *hidden))
            .reshape(32, -1)
        )
        torch.testing.assert_close(actual[:, 0], expected, atol=1e-11, rtol=0)
        assert float(actual[:, 1:].abs().square().sum()) < 1e-22


@qfunc
def computed_selector_program(q: Output[Annotated[QNum, 4]]) -> None:
    allocate(q)
    control(q[0:2] == 0, lambda: payload(0.37, -0.23, q[3:4]))


def test_computed_predicates_retain_whole_register_protection() -> None:
    with pytest.raises(HybridVerificationError, match="disjoint"):
        synth.synthesize(computed_selector_program)


@qfunc
def const_laundering_helper(q: Const[QNum]) -> None:
    control(q[0] == 1, lambda: payload(0.37, -0.23, q[1:2]))


@qfunc
def const_laundering_program(q: Output[Annotated[QNum, 4]]) -> None:
    allocate(q)
    const_laundering_helper(q)


def test_disjoint_callback_does_not_launder_const_storage() -> None:
    with pytest.raises(HybridVerificationError, match="Const|writable|borrow|control"):
        synth.synthesize(const_laundering_program)
