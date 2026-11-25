"""Raw-word tables retain all columns, phase signs and inverse composition."""

import math

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration
ANGLES = (0.73, -1.21, 2 * math.pi, -2 * math.pi)


def rotation(axis: str, angle: float) -> torch.Tensor:
    identity = torch.eye(2, dtype=torch.complex128)
    pauli = torch.tensor(
        (
            [[0, 1], [1, 0]]
            if axis == "X"
            else [[0, -1j], [1j, 0]] if axis == "Y" else [[1, 0], [0, -1]]
        ),
        dtype=torch.complex128,
    )
    return math.cos(angle / 2) * identity - 1j * math.sin(angle / 2) * pauli


def reference(
    axis: str, word: int | None, target: int, *, inverse: bool = False
) -> fq.Circuit:
    circuit = fq.Circuit(3, dtype=torch.complex128)
    if word is None:
        circuit.h(0).h(1).h(2).s(2)
    else:
        for wire in range(2):
            if word & (1 << (1 - wire)):
                circuit.x(wire)
        if target:
            circuit.x(2)
    matrix = torch.zeros((8, 8), dtype=torch.complex128)
    for index, angle in enumerate(ANGLES):
        matrix[2 * index : 2 * index + 2, 2 * index : 2 * index + 2] = rotation(
            axis, -angle if inverse else angle
        )
    return circuit.any(0, 1, 2, matrix=matrix)


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
@pytest.mark.parametrize("word", [0, 1, 2, 3])
@pytest.mark.parametrize("target", [0, 1])
def test_all_raw_signed_fractional_columns_and_conditional_global_sign(
    axis: str, word: int, target: int
) -> None:
    vector = [float(i == word) for i in range(4)]
    source = definition(
        "p:Output[QNum[2,True,2]],q:Output[QBit]",
        f"prepare_amplitudes({vector!r},0,p)\nallocate(q)\n"
        + ("X(q)\n" if target else "")
        + f"select_rotation('{axis}',{list(ANGLES)!r},p,q)",
    )
    program = synth.synthesize(source, max_workspace=0)
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        fq.run(reference(axis, word, target)).statevector().flatten(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
@pytest.mark.parametrize("inverse", [False, True])
def test_coherent_complex_state_and_adjoint(axis: str, inverse: bool) -> None:
    operation = f"select_rotation('{axis}',{list(ANGLES)!r},p,q)"
    body = "allocate(2,p)\nallocate(q)\nhadamard_transform(p)\nH(q)\nS(q)\n" + (
        "with adjoint():\n    " + operation if inverse else operation
    )
    program = synth.synthesize(
        definition("p:Output[QArray[QBit]],q:Output[QBit]", body), max_workspace=0
    )
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        fq.run(reference(axis, None, 0, inverse=inverse)).statevector().flatten(),
        atol=1e-12,
        rtol=0,
    )


def test_lsb_order_and_static_mutable_helper() -> None:
    helper = "@qfunc\ndef table(p:QArray[QBit],q:QBit):\n    select_rotation('Y',[0.73,-1.21,6.283185307179586,-6.283185307179586],p,q)\n"
    program = synth.synthesize(
        definition(
            "p:Output[QArray[QBit]],q:Output[QBit]",
            "allocate(2,p)\nallocate(q)\nX(p[0])\ntable(p,q)",
            helper,
        ),
        max_workspace=0,
    )
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        fq.run(reference("Y", 1, 0)).statevector().flatten(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    "operation",
    [
        "select_rotation('A',[0,1,2,3],p,q)",
        "select_rotation('Y',[0,1],p,q)",
        "select_rotation('Y',[0,1,2,3],p,p[0])",
        "select_rotation('Y',[0,1,2,q],p,q)",
        "select_rotation('Y',[0,1,2,3],p[0:1],q)",
        "select_rotation('Y',[0,1,2,3],p,unknown)",
    ],
)
def test_axis_shape_alias_dynamic_and_view_refusals(operation: str) -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "p:Output[QArray[QBit]],q:Output[QBit]",
                "allocate(2,p)\nallocate(q)\n" + operation,
            )
        )


def test_qpe_inverse_and_within_restore_entangled_phase_counter() -> None:
    source = definition(
        "p:Output[QNum[2,False,2]],s:Output[QBit],q:Output[QBit]",
        "allocate(p)\nallocate(s)\nallocate(q)\nH(s)\nwith within():\n    qpe(lambda:unitary([[1,0],[0,1j]],s),p)\nwith apply():\n    select_rotation('Y',[0.7,-1.2,6.283185307179586,0.3],p,q)",
    )
    program = synth.synthesize(source, max_workspace=0)
    expected = torch.zeros((4, 2, 2), dtype=torch.complex128)
    expected[0, 0, :] = rotation("Y", 0.7)[:, 0] / math.sqrt(2)
    expected[0, 1, :] = rotation("Y", -1.2)[:, 0] / math.sqrt(2)
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(), expected.flatten(), atol=1e-12, rtol=0
    )


def test_root_adjoint_matrix_qpe_is_the_actual_inverse() -> None:
    source = definition(
        "p:Output[QNum[2,False,2]],s:Output[QBit]",
        "allocate(p)\nallocate(s)\nH(s)\nwith adjoint():\n    qpe(lambda:unitary([[1,0],[0,1j]],s),p)",
    )
    program = synth.synthesize(source, max_workspace=0)
    # QPE adjoint first Fourier-transforms zero, then applies U^-k before Hs.
    data = torch.tensor([[1, 0], [0, 1j]], dtype=torch.complex128)
    c = fq.Circuit(3, dtype=torch.complex128).h(2).h(0).h(1)
    for bit in reversed(range(2)):
        block = torch.zeros((4, 4), dtype=torch.complex128)
        block[:2, :2] = torch.eye(2, dtype=torch.complex128)
        block[2:, 2:] = torch.linalg.matrix_power(data.mH, 1 << bit)
        c.any(1 - bit, 2, matrix=block)
    c.h(0).h(1)
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        fq.run(c).statevector().flatten(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("angle", ["True", "1j", "1e999"])
def test_non_real_or_non_finite_angles_are_refused(angle: str) -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "p:Output[QArray[QBit]],q:Output[QBit]",
                "allocate(1,p)\nallocate(q)\nselect_rotation('Y',[0,"
                + angle
                + "],p,q)",
            )
        )


@pytest.mark.parametrize("setup", ["allocate(2,p)", "allocate(q)"])
def test_uninitialized_operand_is_refused(setup: str) -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "p:Output[QArray[QBit]],q:Output[QBit]",
                setup + "\nselect_rotation('Y',[0,1,2,3],p,q)",
            )
        )


def test_eight_bit_boundary_and_ninth_bit_refusal() -> None:
    for width in (8, 9):
        body = f"allocate({width},p)\nallocate(q)\nX(p[0])\nX(p[{width-1}])\nselect_rotation('Y',{[0.41 if i==129 else -0.32 for i in range(1<<width)]!r},p,q)"
        source = definition("p:Output[QArray[QBit]],q:Output[QBit]", body)
        if width == 9:
            with pytest.raises(HybridVerificationError):
                synth.synthesize(source)
        else:
            program = synth.synthesize(source, max_workspace=0)
            expected = torch.zeros((256, 2), dtype=torch.complex128)
            expected[129] = rotation("Y", 0.41)[:, 0]
            torch.testing.assert_close(
                fq.run(program).statevector().flatten(),
                expected.flatten(),
                atol=1e-12,
                rtol=0,
            )


def test_slice_selects_raw_bits_without_touching_other_storage() -> None:
    body = "allocate(3,p)\nallocate(q)\nX(p[0])\nX(p[1])\nselect_rotation('Z',[0.73,-1.21,6.283185307179586,-6.283185307179586],p[1:3],q)"
    program = synth.synthesize(
        definition("p:Output[QArray[QBit]],q:Output[QBit]", body), max_workspace=0
    )
    expected = torch.zeros(16, dtype=torch.complex128)
    expected[6] = torch.exp(torch.tensor(0.605j, dtype=torch.complex128))
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(), expected, atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    "budget", ["max_qubits", "max_operations", "max_depth", "memory_limit_bytes"]
)
def test_rotation_table_budget_is_enforced(budget: str) -> None:
    from flagquantum.errors import PlanningError

    source = definition(
        "p:Output[QArray[QBit]],q:Output[QBit]",
        "allocate(2,p)\nallocate(q)\nselect_rotation('Y',[0.7,-1.2,0.3,0.4],p,q)",
    )
    with pytest.raises((synth.SynthesisError, HybridVerificationError, PlanningError)):
        synth.synthesize(
            source,
            max_qubits=1 if budget == "max_qubits" else None,
            max_operations=1 if budget == "max_operations" else None,
            max_depth=1 if budget == "max_depth" else None,
            memory_limit_bytes=1 if budget == "memory_limit_bytes" else None,
        )


def test_rotation_marker_is_capture_only() -> None:
    with pytest.raises(TypeError, match="captured"):
        getattr(synth, "select_rotation")("Y", [0.1, 0.2], None, None)
