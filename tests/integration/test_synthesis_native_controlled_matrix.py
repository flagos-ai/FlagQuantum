"""Native matrix control preserves complete unitary columns and conditional phase."""

import math

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration
MATRICES = (
    ((-1, 0), (0, -1)),
    ((0, 1j), (1, 0)),
    ((0, 0, 0, 1j), (1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0)),
)


def independent_controlled(
    matrix: tuple[tuple[complex | int, ...], ...], active: int
) -> torch.Tensor:
    size = len(matrix)
    complete = torch.eye(2 * size, dtype=torch.complex128)
    complete[
        active * size : (active + 1) * size, active * size : (active + 1) * size
    ] = torch.tensor(matrix, dtype=torch.complex128)
    return complete


@pytest.mark.parametrize(
    "matrix,selector,word,active",
    [
        (matrix, selector, word, active)
        for matrix in MATRICES
        for selector in range(2)
        for word in range(len(matrix))
        for active in range(2)
    ],
)
def test_every_column_and_conditional_global_phase(
    matrix: tuple[tuple[complex | int, ...], ...], selector: int, word: int, active: int
) -> None:
    width = len(matrix).bit_length() - 1
    condition = "s" if active else "~s"
    program = synth.synthesize(
        definition(
            f"s:Output[QBit],q:Output[QNum[{width}]]",
            f"s |= {selector}\nq |= {word}\ncontrol({condition},lambda:unitary({matrix!r},q))",
        ),
        max_workspace=0,
    )
    reference = fq.Circuit(1 + width, dtype=torch.complex128)
    for wire in range(1 + width):
        if (selector * len(matrix) + word) & (1 << (width - wire)):
            reference.x(wire)
    reference.any(*range(1 + width), matrix=independent_controlled(matrix, active))
    compare_states(program, reference, ("s", "q"))


def test_coherent_matrix_control_and_adjoint_within() -> None:
    matrix = MATRICES[1]
    helper = f"@qfunc\ndef gated(s:QBit,q:QBit):\n    control(~s,lambda:unitary({matrix!r},q))\n"
    body = "allocate(s)\nallocate(q)\nH(s)\nH(q)\nwith within():\n    gated(s,q)\nwith apply():\n    PHASE(0.37,q)\ngated(s,q)\nwith adjoint():\n    gated(s,q)"
    program = synth.synthesize(
        definition("s:Output[QBit],q:Output[QBit]", body, helper), max_workspace=0
    )
    u = independent_controlled(matrix, 0)
    diagonal = torch.diag(
        torch.tensor(
            [1, complex(math.cos(0.37), math.sin(0.37))] * 2, dtype=torch.complex128
        )
    )
    reference = (
        fq.Circuit(2, dtype=torch.complex128)
        .h(0)
        .h(1)
        .any(0, 1, matrix=u.conj().T @ diagonal @ u)
    )
    compare_states(program, reference, ("s", "q"))


@pytest.mark.parametrize(
    "target,wires", [("q[0:2]", (0, 1, 2)), ("[q[1],q[0]]", (0, 2, 1))]
)
def test_indexed_same_owner_disjoint_bits_preserve_view_order(
    target: str, wires: tuple[int, ...]
) -> None:
    matrix = MATRICES[2]
    program = synth.synthesize(
        definition(
            "q:Output[QArray[QBit]]",
            f"allocate(3,q)\nhadamard_transform(q)\nphase(q[0]+2*q[1]+3*q[2],0.19)\ncontrol(q[2],lambda:unitary({matrix!r},{target}))",
        ),
        max_workspace=0,
    )
    reference = fq.Circuit(3, dtype=torch.complex128)
    for wire, angle in enumerate((0.57, 0.38, 0.19)):
        reference.h(wire).phase(wire, theta=angle)
    reference.any(*wires, matrix=independent_controlled(matrix, 1))
    compare_states(program, reference, ("q",))


@pytest.mark.parametrize(
    "body",
    [
        "allocate(2,q)\ncontrol(q[0],lambda:unitary([[0,1],[1,0]],q[0]))",
        "control(s,lambda:unitary([[0,1],[1,0]],q))",
        "allocate(s)\ncontrol(s,lambda:unitary([[0,1],[1,0]],q))",
        "allocate(s)\nallocate(q)\ncontrol(s<1,lambda:unitary([[0,1],[1,0]],q))",
        "allocate(s)\nallocate(q)\ncontrol(s,lambda:unitary([[1,1],[0,1]],q))",
        "allocate(s)\nallocate(q)\ncontrol(s,lambda:unitary([[1,0],[0]],q))",
    ],
)
def test_alias_initialization_condition_and_matrix_refusals(body: str) -> None:
    ports = (
        "q:Output[QArray[QBit]]" if "q[0]" in body else "s:Output[QBit],q:Output[QBit]"
    )
    with pytest.raises(HybridVerificationError):
        synth.synthesize(definition(ports, body))


def test_multibit_selector_and_combined_width_limit_refuse() -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "s:Output[QNum[2]],q:Output[QBit]",
                "allocate(s)\nallocate(q)\ncontrol(s,lambda:unitary([[0,1],[1,0]],q))",
            )
        )
    matrix = tuple(
        tuple(int(row == column) for column in range(32)) for row in range(32)
    )
    with pytest.raises(HybridVerificationError, match="at most five"):
        synth.synthesize(
            definition(
                "s:Output[QBit],q:Output[QNum[5]]",
                f"allocate(s)\nallocate(q)\ncontrol(s,lambda:unitary({matrix!r},q))",
            )
        )


def test_const_selector_is_not_silently_admitted_as_a_writable_matrix_operand() -> None:
    helper = "@qfunc\ndef gated(s:Const[QBit],q:QBit):\n    control(s,lambda:unitary([[0,1],[1,0]],q))\n"
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "s:Output[QBit],q:Output[QBit]",
                "allocate(s)\nallocate(q)\ngated(s,q)",
                helper,
            )
        )


@pytest.mark.parametrize("constraint", ["width", "operations", "memory"])
def test_actual_resource_budgets_refuse(constraint: str) -> None:
    function = definition(
        "s:Output[QBit],q:Output[QBit]",
        f"allocate(s)\nallocate(q)\nH(s)\ncontrol(s,lambda:unitary({MATRICES[1]!r},q))",
    )
    with pytest.raises((synth.SynthesisError, HybridVerificationError, PlanningError)):
        if constraint == "width":
            synth.synthesize(function, max_qubits=1)
        elif constraint == "operations":
            synth.synthesize(function, max_operations=1)
        else:
            synth.synthesize(function, memory_limit_bytes=16)


def test_five_qubit_combined_matrix_bound_is_executable_and_cleans_work() -> None:
    matrix = tuple(
        tuple((-1 if row == 15 else 1) if row == column else 0 for column in range(16))
        for row in range(16)
    )
    program = synth.synthesize(
        definition(
            "s:Output[QBit],q:Output[QNum[4]]",
            f"allocate(s)\nallocate(q)\nH(s)\nhadamard_transform(q)\ncontrol(s,lambda:unitary({matrix!r},q))",
        )
    )
    reference = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(5):
        reference.h(wire)
    reference.any(*range(5), matrix=independent_controlled(matrix, 1))
    compare_states(program, reference, ("s", "q"))
