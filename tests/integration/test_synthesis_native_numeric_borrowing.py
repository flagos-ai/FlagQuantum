"""Borrowed numeric helpers retain exact formats and verified Const ownership."""

import cmath
import math
from fractions import Fraction
from itertools import product

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_exact_numbers import definition
from tests.synthesis_workflow_helpers import check_state

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("threshold", [-0.5, 0.5, 1.5])
@pytest.mark.parametrize("nested", [False, True])
def test_numeric_const_borrowing_matches_independent_circuit(
    signed: bool, threshold: float, nested: bool
) -> None:
    helper = (
        "@qfunc\ndef rotate(x:Const[QNum],q:QBit):\n"
        f"    control(x>{threshold},lambda:R(0.73,1.5707963267948966,q))\n"
    )
    if nested:
        helper += "@qfunc\ndef payoff(value:Const[QNum],target:QBit):\n    rotate(value,target)\n"
    program = synth.synthesize(
        definition(
            f"x:Output[QNum[3,{signed},1]],q:Output[QBit]",
            "allocate(x)\nallocate(q)\nhadamard_transform(x)\nX(q)\nH(q)\nS(q)\n"
            "phase(1,0.37)\n" + ("payoff(x,q)" if nested else "rotate(x,q)"),
            helper,
        ),
        memory_limit_bytes=2**24,
    )
    matrix = torch.eye(16, dtype=torch.complex128)
    rotation = torch.tensor(
        [
            [math.cos(0.73 / 2), -math.sin(0.73 / 2)],
            [math.sin(0.73 / 2), math.cos(0.73 / 2)],
        ],
        dtype=torch.complex128,
    )
    for raw in range(8):
        value = Fraction(raw - 8 if signed and raw >= 4 else raw, 2)
        if value > Fraction(threshold):
            matrix[2 * raw : 2 * raw + 2, 2 * raw : 2 * raw + 2] = rotation
    reference = fq.Circuit(4, dtype=torch.complex128).h(0).h(1).h(2).x(3).h(3).s(3)
    reference.any(0, 1, 2, 3, matrix=matrix * cmath.exp(0.37j))
    state = fq.run(program).statevector().reshape(16, -1)
    torch.testing.assert_close(
        state[:, 0], fq.run(reference).statevector().flatten(), atol=1e-12, rtol=0
    )
    assert float(state[:, 1:].abs().square().sum()) < 1e-22


@pytest.mark.parametrize("inverse", [False, True])
def test_helper_numeric_predicate_phase_and_inverse_in_body(inverse: bool) -> None:
    operation = "control((x>0.5)&(x<1.5),lambda:phase(1,0.73))"
    if inverse:
        operation = "with adjoint():\n        " + operation
    helper = "@qfunc\ndef payoff(x:Const[QNum],q:QBit):\n    " + operation + "\n"
    program = synth.synthesize(
        definition(
            "x:Output[QNum[3,True,1]],q:Output[QBit]",
            "allocate(x)\nallocate(q)\nhadamard_transform(x)\nH(q)\npayoff(x,q)",
            helper,
        )
    )
    expected = {}
    for raw, q in product(range(8), range(2)):
        value = Fraction(raw - 8 if raw >= 4 else raw, 2)
        angle = (
            (-0.73 if inverse else 0.73)
            if Fraction(1, 2) < value < Fraction(3, 2)
            else 0
        )
        expected[(raw, q)] = cmath.exp(1j * angle) / 4
    check_state(program, fq.run(program), expected)


@pytest.mark.parametrize(
    "effect",
    [
        "X(x)",
        "H(x)",
        "R(0.73,0.5,x)",
        "PHASE(0.73,x)",
        "CX(q,x)",
        "SWAP(q,x)",
        "CRX(0.73,q,x)",
        "control(q,lambda:X(x))",
        "x ^= 1",
        "drop(x)",
    ],
)
def test_const_formal_mutation_is_rejected_by_effect_verifier(effect: str) -> None:
    helper = "@qfunc\ndef bad(x:Const[QBit],q:QBit):\n    " + effect + "\n"
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "x:Output[QBit],q:Output[QBit]",
                "allocate(x)\nallocate(q)\nbad(x,q)",
                helper,
            )
        )


def test_const_cannot_be_laundered_through_nested_mutable_helper() -> None:
    helper = (
        "@qfunc\ndef mutate(a:QBit):\n    X(a)\n"
        "@qfunc\ndef bad(x:Const[QBit],q:QBit):\n    mutate(x)\n"
    )
    with pytest.raises(HybridVerificationError, match="Const"):
        synth.synthesize(
            definition(
                "x:Output[QBit],q:Output[QBit]",
                "allocate(x)\nallocate(q)\nbad(x,q)",
                helper,
            )
        )


@pytest.mark.parametrize("arguments", ["x,x", "q,q"])
def test_readonly_and_writable_actuals_cannot_overlap(arguments: str) -> None:
    helper = "@qfunc\ndef copy(x:Const[QBit],q:QBit):\n    CX(x,q)\n"
    with pytest.raises(HybridVerificationError, match="disjoint"):
        synth.synthesize(
            definition(
                "x:Output[QBit],q:Output[QBit]",
                "allocate(x)\nallocate(q)\n" + f"copy({arguments})",
                helper,
            )
        )


@pytest.mark.parametrize(
    "body",
    [
        "allocate(q)\npayoff(x,q)",
        "allocate(x)\nallocate(q)\ny:QNum[3,False,1]=QNum()\nallocate(y)\ndrop(y)\npayoff(y,q)",
    ],
)
def test_helper_borrowing_requires_live_initialized_actuals(body: str) -> None:
    helper = (
        "@qfunc\ndef payoff(x:Const[QNum],q:QBit):\n    control(x>0.5,lambda:X(q))\n"
    )
    with pytest.raises(HybridVerificationError, match="uninitialized|dropped"):
        synth.synthesize(
            definition("x:Output[QNum[3,False,1]],q:Output[QBit]", body, helper)
        )


def test_explicit_helper_format_cannot_discard_fraction_bits() -> None:
    helper = (
        "@qfunc\ndef payoff(x:Const[QNum[3]],q:QBit):\n    control(x>0,lambda:X(q))\n"
    )
    with pytest.raises(HybridVerificationError, match="formats"):
        synth.synthesize(
            definition(
                "x:Output[QNum[3,False,1]],q:Output[QBit]",
                "allocate(x)\nallocate(q)\npayoff(x,q)",
                helper,
            )
        )


def test_unsized_numeric_mutable_helper_preserves_the_actual_word() -> None:
    helper = "@qfunc\ndef transform(a:QNum):\n    hadamard_transform(a)\n"
    program = synth.synthesize(
        definition(
            "x:Output[QNum[3,True,1]]", "allocate(x)\nx ^= 1\ntransform(x)", helper
        )
    )
    reference = fq.Circuit(3, dtype=torch.complex128).x(2).h(0).h(1).h(2)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


def test_explicit_helper_width_cannot_be_respecialized() -> None:
    helper = (
        "@qfunc\ndef payoff(x:Const[QNum[2]],q:QBit):\n    control(x>0,lambda:X(q))\n"
    )
    with pytest.raises(HybridVerificationError, match="width"):
        synth.synthesize(
            definition(
                "x:Output[QNum[3]],q:Output[QBit]",
                "allocate(x)\nallocate(q)\npayoff(x,q)",
                helper,
            )
        )


@pytest.mark.parametrize("constraint", ["workspace", "qubits", "operations", "memory"])
def test_borrowed_predicates_charge_actual_resource_budgets(constraint: str) -> None:
    helper = "@qfunc\ndef payoff(x:Const[QNum],q:QBit):\n    control(x>0.5,lambda:R(0.73,1.5707963267948966,q))\n"
    entry = definition(
        "x:Output[QNum[3,True,1]],q:Output[QBit]",
        "allocate(x)\nallocate(q)\nhadamard_transform(x)\npayoff(x,q)",
        helper,
    )
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if constraint == "workspace":
            synth.synthesize(entry, max_workspace=0)
        elif constraint == "qubits":
            synth.synthesize(entry, max_qubits=4)
        elif constraint == "operations":
            synth.synthesize(entry, max_operations=1)
        else:
            synth.synthesize(entry, memory_limit_bytes=16)
