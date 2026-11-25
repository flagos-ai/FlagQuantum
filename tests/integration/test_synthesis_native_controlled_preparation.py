"""Controlled preparation and complete nonuniform quantum walk acceptance."""

from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.basics.quantumwalk_complex_network import main as walk
from examples.synthesis.tutorials.basics.quantumwalk_complex_network.comparisons.circuit import (
    reference_circuit,
)
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/tutorials/basic_tutorials/quantumwalk_complex_network/quantumwalk_complex_network.qmod"
)


def check(program: synth.Program, expected: torch.Tensor) -> None:
    state = fq.run(program).statevector().reshape(16, -1)
    torch.testing.assert_close(state[:, 0], expected.flatten(), atol=1e-12, rtol=0)
    assert float(state[:, 1:].abs().square().sum()) < 1e-22


@pytest.mark.parametrize("steps", [0, 1, 2, 3])
@pytest.mark.parametrize("angle", [0.0, 0.37, -0.61])
def test_every_round_complete_environment_and_nonzero_phases(
    steps: int, angle: float
) -> None:
    check(
        walk.build_program(steps, angle),
        fq.run(reference_circuit(steps, angle)).statevector(),
    )


def test_frozen_original_native_and_independent_absolute_full_state() -> None:
    expected = fq.run(reference_circuit()).statevector()
    check(synth.synthesize(SOURCE), expected)
    check(walk.build_program(), expected)


@pytest.mark.parametrize(
    "body",
    [
        "control(x, lambda: inplace_prepare_state([0.5,0.5],0,x))",
        "control(x, lambda: allocate(y))",
        "control(x, lambda: publish(y))",
        "control(True, lambda: inplace_prepare_state([0.5,0.5],0,y))",
        "control(False, lambda: inplace_prepare_state([0.5,0.5],0,y))",
    ],
)
def test_overlap_allocation_and_output_publication_refuse(body: str) -> None:
    helper = "@qfunc\ndef publish(y:Output[QBit]):\n    allocate(y)\n"
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "x:Output[QBit],y:Output[QBit]",
                "allocate(x)\nallocate(y)\n" + body,
                helper,
            )
        )


@pytest.mark.parametrize("selected", [0, 1])
@pytest.mark.parametrize("probability,angle", [(0.2, 0.37), (0.7, -0.61)])
@pytest.mark.parametrize("inverse", [False, True])
def test_controlled_helper_complex_target_and_conditional_global_phase(
    selected: int, probability: float, angle: float, inverse: bool
) -> None:
    import math

    statements = (
        f"inplace_prepare_state([{probability},{1-probability}],0,q)\n"
        f"phase(1,{angle})"
    )
    if inverse:
        statements = "with adjoint():\n" + "\n".join(
            "    " + line for line in statements.splitlines()
        )
    helper = (
        "@qfunc\ndef twist(q:QBit):\n"
        + "\n".join("    " + line for line in statements.splitlines())
        + "\n"
    )
    body = (
        "allocate(c)\nallocate(t)\nhadamard_transform(c)\nhadamard_transform(t)\n"
        "phase(c,0.11)\nphase(t,0.23)\n"
        f"control(c=={selected},lambda:twist(t))"
    )
    program = synth.synthesize(
        definition("c:Output[QBit],t:Output[QBit]", body, helper)
    )
    rotation = torch.tensor(
        [
            [math.sqrt(probability), -math.sqrt(1 - probability)],
            [math.sqrt(1 - probability), math.sqrt(probability)],
        ],
        dtype=torch.complex128,
    ) * torch.exp(torch.tensor(1j * angle, dtype=torch.complex128))
    if inverse:
        rotation = rotation.conj().T
    matrix = torch.eye(4, dtype=torch.complex128)
    matrix[2 * selected : 2 * selected + 2, 2 * selected : 2 * selected + 2] = rotation
    reference = (
        fq.Circuit(2, dtype=torch.complex128)
        .h(0)
        .h(1)
        .phase(0, theta=0.11)
        .phase(1, theta=0.23)
        .any(0, 1, matrix=matrix)
    )
    actual = fq.run(program).statevector().reshape(4, -1)
    torch.testing.assert_close(
        actual[:, 0], fq.run(reference).statevector().flatten(), atol=1e-12, rtol=0
    )
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22


@pytest.mark.parametrize(
    "kind", ["qubits", "workspace", "operations", "depth", "memory"]
)
def test_actual_walk_resource_budgets_refuse(kind: str) -> None:
    from flagquantum.errors import PlanningError

    inputs = {"steps": 3, "angle": 0.37}
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if kind == "qubits":
            synth.synthesize(walk.main, classical_inputs=inputs, max_qubits=4)
        elif kind == "workspace":
            synth.synthesize(walk.main, classical_inputs=inputs, max_workspace=0)
        elif kind == "operations":
            synth.synthesize(walk.main, classical_inputs=inputs, max_operations=1)
        elif kind == "depth":
            synth.synthesize(walk.main, classical_inputs=inputs, max_depth=1)
        else:
            synth.synthesize(walk.main, classical_inputs=inputs, memory_limit_bytes=16)


def test_unused_callback_argument_cannot_alias_condition() -> None:
    helper = "@qfunc\ndef scalar_only(q:QBit):\n    phase(1,0.37)\n"
    with pytest.raises(HybridVerificationError, match="disjoint"):
        synth.synthesize(
            definition(
                "c:Output[QBit]",
                "allocate(c)\ncontrol(c,lambda:scalar_only(c))",
                helper,
            )
        )


def test_nested_output_escape_refuses() -> None:
    helper = (
        "@qfunc\ndef publish(q:Output[QBit]):\n    allocate(q)\n"
        "@qfunc\ndef escape(q:QBit):\n    publish(q)\n"
    )
    with pytest.raises(HybridVerificationError, match="Output"):
        synth.synthesize(
            definition(
                "c:Output[QBit],t:Output[QBit]",
                "allocate(c)\nallocate(t)\ncontrol(c,lambda:escape(t))",
                helper,
            )
        )


def test_const_target_refuses() -> None:
    helper = "@qfunc\ndef protected(c:QBit,t:QBit):\n    control(c,lambda:inplace_prepare_state([0.5,0.5],0,t))\n"
    with pytest.raises(HybridVerificationError, match="writable|const"):
        synth.synthesize(
            definition(
                "c:Output[QBit],t:Const[QBit]", "allocate(c)\nprotected(c,t)", helper
            )
        )
