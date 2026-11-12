"""Captured controlled Pauli rotations retain complex phase and clean predicates."""

import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.basics.quantum_programming.controlled_rotation.comparisons.circuit import (
    reference_circuit as tutorial_reference,
)
from examples.synthesis.tutorials.basics.quantum_programming.controlled_rotation.main import (
    build_program,
)
from flagquantum import synth
from flagquantum.compiler._construction.frontend_python import _QuantumFunction
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.compiler.resource_estimation import ResourceEstimate
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_polynomial_phase import definition

pytestmark = pytest.mark.integration
ERRORS = (synth.SynthesisError, HybridVerificationError)
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/tutorials/basic_tutorials/the_classiq_tutorial/Qmod_tutorial_part2.qmod"
)


def controlled_reference(theta: float, phi: float) -> torch.Tensor:
    """Build the exact block matrix without using any Compiler decomposition."""
    cosine = math.cos(theta / 2)
    sine = math.sin(theta / 2)
    matrix = torch.eye(4, dtype=torch.complex128)
    lower = complex(math.cos(phi), math.sin(phi))
    matrix[2:, 2:] = torch.tensor(
        [[cosine, -1j * sine * lower.conjugate()], [-1j * sine * lower, cosine]],
        dtype=torch.complex128,
    )
    return matrix


@pytest.mark.parametrize("phi", [0.0, math.pi / 2, math.pi, -math.pi / 2])
@pytest.mark.parametrize("preparation", ["zero", "one", "coherent"])
def test_axes_and_both_control_branches_match_independent_matrix(
    phi: float,
    preparation: str,
) -> None:
    body = "allocate(c)\nallocate(q)\nhadamard_transform(q)\nPHASE(0.23,q)\n"
    reference = fq.Circuit(2, dtype=torch.complex128).h(1).phase(1, theta=0.23)
    if preparation == "one":
        body += "X(c)\n"
        reference.x(0)
    elif preparation == "coherent":
        body += "H(c)\nPHASE(0.41,c)\n"
        reference.h(0).phase(0, theta=0.41)
    body += f"control(c,lambda:R(-0.73,{phi!r},q))"
    program = synth.synthesize(
        definition(body, "c:Output[QBit],q:Output[QBit]"), max_workspace=0
    )
    reference.any(0, 1, matrix=controlled_reference(-0.73, phi))
    compare_states(program, reference, ("c", "q"))
    assert program.report["workspace_qubits"] == 0


def test_original_tutorial_and_native_match_complete_state_and_cleanup() -> None:
    reference = tutorial_reference()
    native = build_program()
    original = synth.synthesize(SOURCE)
    compare_states(native, reference, ("msb", "lsb"))
    compare_states(original, reference, ("res",))
    workspace = native.report["workspace_qubits"]
    assert isinstance(workspace, int) and workspace > 0
    resources = native.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits == 3 + workspace


@pytest.mark.parametrize(
    "body,signature,diagnostic",
    [
        (
            "allocate(c)\nallocate(q)\ncontrol(c,lambda:R(0.2,0.3,q))",
            "c:Output[QBit],q:Output[QBit]",
            "Pauli X or Y",
        ),
        ("allocate(c)\ncontrol(c,lambda:R(0.2,0.0,c))", "c:Output[QBit]", "disjoint"),
        (
            "allocate(c)\ncontrol(c,lambda:R(0.2,0.0,q))",
            "c:Output[QBit],q:Output[QBit]",
            "initialized",
        ),
        (
            "allocate(q)\ncontrol(c,lambda:R(0.2,0.0,q))",
            "c:Output[QBit],q:Output[QBit]",
            "initialized",
        ),
        (
            "allocate(c)\ncontrol(c<2,lambda:R(0.2,0.0,q))",
            "c:Output[QNum[9]],q:Output[QBit]",
            "eight input bits",
        ),
    ],
)
def test_invalid_rotation_storage_and_predicates_fail_closed(
    body: str,
    signature: str,
    diagnostic: str,
) -> None:
    with pytest.raises(ERRORS, match=diagnostic):
        synth.synthesize(definition(body, signature))


@pytest.mark.parametrize(
    "constraint",
    ["max_qubits", "max_operations", "memory_limit_bytes"],
)
def test_controlled_rotation_obeys_actual_resource_limits(constraint: str) -> None:
    entry: _QuantumFunction = definition(
        "allocate(c)\nallocate(q)\nH(c)\ncontrol(c,lambda:R(0.73,0.0,q))",
        "c:Output[QBit],q:Output[QBit]",
    )
    with pytest.raises((*ERRORS, PlanningError)):
        if constraint == "max_qubits":
            synth.synthesize(entry, max_qubits=1)
        elif constraint == "max_operations":
            synth.synthesize(entry, max_operations=1)
        else:
            synth.synthesize(entry, memory_limit_bytes=16)


@pytest.mark.parametrize("phi", [math.pi / 2 + 1e-13, math.pi / 2 - 1e-13])
def test_near_axis_angles_are_rejected_without_silent_rounding(phi: float) -> None:
    with pytest.raises(ERRORS, match="Pauli X or Y"):
        synth.synthesize(
            definition(
                f"allocate(c)\nallocate(q)\ncontrol(c,lambda:R(0.2,{phi!r},q))",
                "c:Output[QBit],q:Output[QBit]",
            )
        )


def test_multibit_register_is_not_silently_treated_as_its_low_bit() -> None:
    with pytest.raises(ERRORS, match="one-bit condition"):
        synth.synthesize(
            definition(
                "allocate(c)\nallocate(q)\ncontrol(c,lambda:R(0.2,0.0,q))",
                "c:Output[QNum[2]],q:Output[QBit]",
            )
        )


def test_const_target_cannot_be_mutated_by_controlled_rotation() -> None:
    entry = definition(
        "allocate(c)\nallocate(q)\nbad(c,q)",
        "c:Output[QBit],q:Output[QBit]",
        "@qfunc\ndef bad(c:Const[QBit],q:Const[QBit]):\n    control(c,lambda:R(0.2,0.0,q))\n",
    )
    with pytest.raises(ERRORS, match="writable"):
        synth.synthesize(entry)


@pytest.mark.parametrize("index", [0, 2])
def test_indexed_target_retains_array_bit_order_and_spectators(index: int) -> None:
    entry = definition(
        "allocate(c)\nallocate(3,q)\nH(c)\nhadamard_transform(q)\n"
        f"PHASE(0.29,q[1])\ncontrol(c,lambda:R(0.37,{math.pi / 2!r},q[{index}]))",
        "c:Output[QBit],q:Output[QArray[QBit]]",
    )
    program = synth.synthesize(entry, max_workspace=0)
    reference = fq.Circuit(4, dtype=torch.complex128)
    for wire in range(4):
        reference.h(wire)
    reference.phase(2, theta=0.29)
    reference.any(0, 3 - index, matrix=controlled_reference(0.37, math.pi / 2))
    compare_states(program, reference, ("c", "q"))
    assert program.report["workspace_qubits"] == 0
