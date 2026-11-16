"""Joint operations preserve named storage, explicit bit order and every column."""

import math

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_exact_numbers import definition
from tests.integration.test_synthesis_native_inplace_preparation import (
    independent_unitary,
)

pytestmark = pytest.mark.integration
VECTOR = (0.5, -0.5, 0.5, 0.5)


@pytest.mark.parametrize("word", range(4))
@pytest.mark.parametrize("probabilities", [False, True])
def test_joint_signed_fractional_storage_all_input_columns(
    word: int, probabilities: bool
) -> None:
    values = tuple(v * v for v in VECTOR) if probabilities else VECTOR
    operation = (
        "inplace_prepare_state" if probabilities else "inplace_prepare_amplitudes"
    )
    source = definition(
        "a:Output[QBit],b:Output[QNum[1,True,1]]",
        f"a |= {word&1}\nallocate(b)\nb ^= {word>>1}\n{operation}({list(values)!r},0,(a,b))",
    )
    program = synth.synthesize(source, max_workspace=0)
    vector = tuple(math.sqrt(v) for v in values) if probabilities else VECTOR
    # Explicit sequence a,b means a is the low logical bit; physical a is first.
    c = fq.Circuit(2, dtype=torch.complex128)
    if word & 1:
        c.x(0)
    if word >> 1:
        c.x(1)
    c.any(1, 0, matrix=independent_unitary(vector))
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        fq.run(c).statevector().flatten(),
        atol=1e-12,
        rtol=0,
    )
    assert dict(program.registers) == {"a": (0,), "b": (1,)}


def test_joint_array_slice_with_entangled_spectator_and_inverse() -> None:
    prep = f"inplace_prepare_amplitudes({list(VECTOR)!r},0,(a,b[1:2]))"
    source = definition(
        "a:Output[QBit],b:Output[QArray[QBit]],ref:Output[QBit]",
        "allocate(a)\nallocate(2,b)\nallocate(ref)\nH(ref)\nCX(ref,a)\nX(b[0])\n"
        + prep
        + "\nwith adjoint():\n    "
        + prep,
    )
    program = synth.synthesize(source, max_workspace=0)
    c = fq.Circuit(4, dtype=torch.complex128).h(3).cx(3, 0).x(2)
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        fq.run(c).statevector().flatten(),
        atol=1e-12,
        rtol=0,
    )


def test_joint_mutable_helper_and_within_absolute_phase() -> None:
    helper = "@qfunc\ndef prepare(a:QBit,b:QNum):\n    inplace_prepare_amplitudes([0.5,-0.5,0.5,0.5],0,(a,b))\n"
    source = definition(
        "a:Output[QBit],b:Output[QNum[1]]",
        "allocate(a)\nallocate(b)\nH(a)\nS(a)\nH(b)\nwith within():\n    prepare(a,b)\nwith apply():\n    phase(a*b,0.41)",
        helper,
    )
    program = synth.synthesize(source, max_workspace=3)
    u = independent_unitary(VECTOR)
    d = torch.diag(
        torch.tensor(
            [1, 1, 1, complex(math.cos(0.41), math.sin(0.41))], dtype=torch.complex128
        )
    )
    c = (
        fq.Circuit(5, dtype=torch.complex128)
        .h(0)
        .s(0)
        .h(1)
        .any(1, 0, matrix=u.mH @ d @ u)
    )
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        fq.run(c).statevector().flatten(),
        atol=1e-12,
        rtol=0,
    )

    assert program.report["workspace_qubits"] == 3
    assert (
        float(fq.run(program).statevector().reshape(-1, 8)[:, 1:].abs().square().sum())
        < 1e-24
    )


def test_joint_whole_register_matrix_targets_preserve_numeric_labels() -> None:
    source = definition(
        "a:Output[QBit],b:Output[QNum[2]]",
        "allocate(a)\nallocate(b)\nH(a)\nunitary([[0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0]],(a,b))",
    )
    program = synth.synthesize(source)
    assert dict(
        program.probabilities(fq.run(program), registers=("a", "b"))
    ) == pytest.approx({(1, 0): 0.5, (0, 1): 0.5})


@pytest.mark.parametrize(
    "target",
    ["()", "(a,a)", "((a,b),)", "(a,b,a)", "(a,unknown)", "(b[2],a)", "(b[1:0],a)"],
)
def test_bad_joint_target_refusals(target: str) -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "a:Output[QBit],b:Output[QArray[QBit]]",
                f"allocate(a)\nallocate(2,b)\ninplace_prepare_amplitudes([0.5,0.5,0.5,0.5],0,{target})",
            )
        )


@pytest.mark.parametrize("setup", ["allocate(a)", "allocate(b)"])
def test_every_joint_owner_must_be_initialized(setup: str) -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "a:Output[QBit],b:Output[QNum[1]]",
                setup + "\ninplace_prepare_amplitudes([0.5,0.5,0.5,0.5],0,(a,b))",
            )
        )


def test_joint_borrowing_cannot_mutate_const_or_alias_formals() -> None:
    for signature, actual in [
        ("a:Const[QNum[1]],b:QBit", "a,b"),
        ("a:QBit,b:QBit", "a,a"),
    ]:
        helper = f"@qfunc\ndef bad({signature}):\n    inplace_prepare_amplitudes([0.5,0.5,0.5,0.5],0,(a,b))\n"
        with pytest.raises(HybridVerificationError):
            synth.synthesize(
                definition(
                    "a:Output[QBit],b:Output[QBit]",
                    f"allocate(a)\nallocate(b)\nbad({actual})",
                    helper,
                )
            )


@pytest.mark.parametrize("budget", ["width", "operations", "depth", "memory"])
def test_joint_actual_budget_refusals(budget: str) -> None:
    source = definition(
        "a:Output[QBit],b:Output[QNum[1]]",
        "allocate(a)\nallocate(b)\ninplace_prepare_amplitudes([0.5,-0.5,0.5,0.5],0,(a,b))",
    )
    with pytest.raises((HybridVerificationError, synth.SynthesisError, PlanningError)):
        synth.synthesize(
            source,
            max_qubits=1 if budget == "width" else None,
            max_operations=1 if budget == "operations" else None,
            max_depth=1 if budget == "depth" else None,
            memory_limit_bytes=1 if budget == "memory" else None,
        )


def test_joint_eight_bit_preparation_boundary_and_spectator_storage() -> None:
    vector = [0.0] * 256
    vector[193] = -1.0
    source = definition(
        "a:Output[QNum[4]],b:Output[QNum[4]],spectator:Output[QBit]",
        f"allocate(a)\nallocate(b)\nallocate(spectator)\nH(spectator)\ninplace_prepare_amplitudes({vector!r},0,(a,b))",
    )
    program = synth.synthesize(source, max_workspace=0)
    expected = torch.zeros((16, 16, 2), dtype=torch.complex128)
    expected[1, 12, :] = -1 / math.sqrt(2)
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(), expected.flatten(), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    "operation",
    [
        "inplace_prepare_amplitudes(" + repr([1.0] + [0.0] * 511) + ",0,(a,b))",
        "unitary([[1,0],[0,1]],(a,b))",
    ],
)
def test_joint_width_limits_reject_without_truncation(operation: str) -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "a:Output[QNum[5]],b:Output[QNum[4]]",
                "allocate(a)\nallocate(b)\n" + operation,
            )
        )
