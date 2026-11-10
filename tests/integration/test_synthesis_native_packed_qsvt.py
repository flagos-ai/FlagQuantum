"""Packed QSVT originals, exact projected phases and clean borrowed inverses."""

import math
from pathlib import Path
from typing import Annotated

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.cfd.heat_eq_qsvt import main as heat
from examples.synthesis.applications.cfd.heat_eq_qsvt.comparisons.circuit import (
    reference_circuit as heat_reference,
)
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt import (
    main as paper,
)
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.comparisons.circuit import (
    reference_circuit as paper_reference,
)
from flagquantum import synth
from flagquantum.errors import PlanningError
from flagquantum.synth import (
    H,
    Output,
    QBit,
    QNum,
    adjoint,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)

pytestmark = pytest.mark.integration
ORIGINALS = (
    Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod/corpus/original"
)


def verify(program: synth.Program, expected: torch.Tensor) -> torch.Tensor:
    raw = fq.run(program).statevector()
    assert isinstance(raw, torch.Tensor)
    state = raw.reshape(expected.numel(), -1)
    torch.testing.assert_close(state[:, 0], expected.flatten(), atol=1e-10, rtol=0)
    assert float(state[:, 1:].abs().square().sum()) < 1e-22
    return state[:, 0]


@pytest.mark.parametrize("delta", [0.0, 0.37, -0.61])
def test_paper_full_states_phases_and_source_packing(delta: float) -> None:
    phases = tuple(angle + delta * (i + 1) / 4 for i, angle in enumerate(paper.PHASES))
    expected = fq.run(paper_reference(phases)).statevector()
    program = paper.build_program(phases)
    actual = verify(program, expected)
    assert len(program.registers["data_and_deletion"]) == 7
    if delta == 0:
        original = synth.synthesize(
            ORIGINALS
            / "tutorials/technology_demonstrations/classiq_paper/qsvt/classiq_qsvt.qmod"
        )
        raw = fq.run(original).statevector().reshape(1024, -1)
        reordered = raw[:, 0].reshape(2, 2, 2, 64, 2).permute(1, 3, 2, 0, 4).flatten()
        torch.testing.assert_close(actual, reordered, atol=1e-10, rtol=0)
        assert float(raw[:, 1:].abs().square().sum()) < 1e-22


@pytest.mark.parametrize("delta", [0.0, 0.37, -0.61])
def test_heat_full_states_phases_and_source_packing(delta: float) -> None:
    phases = tuple(angle + delta * (i + 1) / 4 for i, angle in enumerate(heat.PHASES))
    expected = fq.run(heat_reference(phases)).statevector()
    program = heat.build_program(phases)
    actual = verify(program, expected)
    assert len(program.registers["data_and_deletion"]) == 4
    if delta == 0:
        original = synth.synthesize(
            ORIGINALS / "applications/CFD/heat_eq_qsvt/heat_eq_qsvt.qmod"
        )
        raw = fq.run(original).statevector().reshape(256, -1)
        reordered = raw[:, 0].reshape(2, 2, 4, 8, 2).permute(0, 3, 2, 1, 4).flatten()
        torch.testing.assert_close(actual, reordered, atol=1e-10, rtol=0)
        assert float(raw[:, 1:].abs().square().sum()) < 1e-22


@pytest.mark.parametrize(
    "rhs",
    [
        (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
        (-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0),
    ],
)
def test_heat_nonzero_input_vectors(rhs: tuple[float, ...]) -> None:
    norm = math.sqrt(sum(x * x for x in rhs))
    amplitudes = tuple(x / norm for x in rhs)
    verify(
        heat.build_program(amplitudes=amplitudes),
        fq.run(heat_reference(amplitudes=amplitudes)).statevector(),
    )


def test_heat_preserves_unused_fourth_phase_behavior() -> None:
    changed = (*heat.PHASES[:3], heat.PHASES[3] + 1.37)
    expected = fq.run(heat_reference()).statevector()
    verify(heat.build_program(changed), expected)


@qfunc
def paper_roundtrip(
    packed: Output[Annotated[QNum, 7]], select: Output[QBit], projection: Output[QBit]
) -> None:
    allocate(packed)
    allocate(select)
    allocate(projection)
    hadamard_transform(packed)
    H(select)
    H(projection)
    phase(packed, 0.13)
    paper.be(packed, select, projection)
    with adjoint():
        paper.be(packed, select, projection)


@qfunc
def heat_roundtrip(
    packed: Output[Annotated[QNum, 4]],
    select: Output[Annotated[QNum, 2]],
    rotation: Output[QBit],
) -> None:
    allocate(packed)
    allocate(select)
    allocate(rotation)
    hadamard_transform(packed)
    hadamard_transform(select)
    H(rotation)
    phase(packed, 0.13)
    heat.be(packed, select, rotation)
    with adjoint():
        heat.be(packed, select, rotation)


@pytest.mark.parametrize("kind", ["paper", "heat"])
def test_checked_block_inverse_on_complete_complex_uniform_state(kind: str) -> None:
    width, packed = (9, 7) if kind == "paper" else (7, 4)
    reference = fq.Circuit(width, dtype=torch.complex128)
    for wire in range(width):
        reference.h(wire)
    for bit in range(packed):
        reference.phase(packed - 1 - bit, theta=0.13 * (1 << bit))
    program = synth.synthesize(
        paper_roundtrip if kind == "paper" else heat_roundtrip,
        memory_limit_bytes=64 * 1024**2,
    )
    verify(program, fq.run(reference).statevector())


@pytest.mark.parametrize("kind", ["paper", "heat"])
@pytest.mark.parametrize(
    "limit", ["qubits", "workspace", "operations", "depth", "memory"]
)
def test_resource_budgets(kind: str, limit: str) -> None:
    entry = paper.main if kind == "paper" else heat.main
    inputs = {"phases": list(paper.PHASES if kind == "paper" else heat.PHASES)}
    if kind == "heat":
        inputs["amplitudes"] = list(heat.AMPLITUDES) + [0.0] * 8
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if limit == "qubits":
            synth.synthesize(
                entry, classical_inputs=inputs, max_qubits=9 if kind == "paper" else 7
            )
        elif limit == "workspace":
            synth.synthesize(entry, classical_inputs=inputs, max_workspace=0)
        elif limit == "operations":
            synth.synthesize(entry, classical_inputs=inputs, max_operations=1)
        elif limit == "depth":
            synth.synthesize(entry, classical_inputs=inputs, max_depth=1)
        else:
            synth.synthesize(entry, classical_inputs=inputs, memory_limit_bytes=16)


@pytest.mark.parametrize("kind", ["paper", "heat"])
def test_incomplete_phase_array_refuses(kind: str) -> None:
    with pytest.raises(ValueError, match="four phases"):
        if kind == "paper":
            paper.build_program(paper.PHASES[:3])
        else:
            heat.build_program(heat.PHASES[:3])
