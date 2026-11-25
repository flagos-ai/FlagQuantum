"""Read-only numeric callbacks preserve joint amplitudes and clean scratch."""

import cmath
import math
from fractions import Fraction

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("threshold", [-0.5, 0.5, 1.5])
@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("theta", [0.0, 0.73, -1.21])
@pytest.mark.parametrize("phi", [0.0, math.pi / 2, math.pi])
def test_readonly_callback_xor_and_nested_rotation_against_independent_circuit(
    signed: bool, threshold: float, initial: bool, theta: float, phi: float
) -> None:
    helper = (
        "@qfunc\ndef payoff(x:Const[QNum],q:QBit):\n"
        "    q ^= x>1.5\n"
        f"    control(x>0.5,lambda:R({theta!r},{phi!r},q))\n"
        "    phase(1,0.37)\n"
    )
    entry = definition(
        f"x:Output[QNum[3,{signed},1]],q:Output[QBit]",
        "allocate(x)\nallocate(q)\nhadamard_transform(x)\n"
        + ("X(q)\n" if initial else "")
        + f"H(q)\nS(q)\ncontrol(x>{threshold},lambda:payoff(x,q))",
        helper,
    )
    program = synth.synthesize(entry, memory_limit_bytes=2**26)
    matrix = torch.eye(16, dtype=torch.complex128)
    rotation = torch.tensor(
        [
            [math.cos(theta / 2), -1j * cmath.exp(-1j * phi) * math.sin(theta / 2)],
            [-1j * cmath.exp(1j * phi) * math.sin(theta / 2), math.cos(theta / 2)],
        ],
        dtype=torch.complex128,
    )
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    for raw in range(8):
        value = Fraction(raw - 8 if signed and raw >= 4 else raw, 2)
        if value <= Fraction(threshold):
            continue
        block = torch.eye(2, dtype=torch.complex128)
        if value > Fraction(3, 2):
            block = x @ block
        if value > Fraction(1, 2):
            block = rotation @ block
        matrix[2 * raw : 2 * raw + 2, 2 * raw : 2 * raw + 2] = cmath.exp(0.37j) * block
    reference = fq.Circuit(4, dtype=torch.complex128).h(0).h(1).h(2)
    if initial:
        reference.x(3)
    reference.h(3).s(3).any(0, 1, 2, 3, matrix=matrix)
    actual = fq.run(program).statevector().reshape(16, -1)
    torch.testing.assert_close(
        actual[:, 0], fq.run(reference).statevector().flatten(), atol=1e-12, rtol=0
    )
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22


@pytest.mark.parametrize("inverse", [False, True])
def test_callback_adjoint_preserves_branch_relative_global_phase(inverse: bool) -> None:
    action = "control(x>0.5,lambda:payoff(x,q))"
    if inverse:
        action = "with adjoint():\n    " + action
    helper = (
        "@qfunc\ndef payoff(x:Const[QNum],q:QBit):\n    q ^= x>1.5\n    phase(1,0.37)\n"
    )
    program = synth.synthesize(
        definition(
            "x:Output[QNum[3,True,1]],q:Output[QBit]",
            "allocate(x)\nallocate(q)\nhadamard_transform(x)\nH(q)\nS(q)\n" + action,
            helper,
        )
    )
    expected = torch.zeros(16, dtype=torch.complex128)
    for raw in range(8):
        value = Fraction(raw - 8 if raw >= 4 else raw, 2)
        for q in range(2):
            target = q ^ int(value > Fraction(3, 2) and value > Fraction(1, 2))
            phase = cmath.exp(
                (-0.37j if inverse else 0.37j) if value > Fraction(1, 2) else 0j
            )
            expected[2 * raw + target] = torch.tensor(
                phase * (1j if q else 1) / 4, dtype=torch.complex128
            )
    actual = fq.run(program).statevector().reshape(16, -1)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22


@pytest.mark.parametrize(
    "helper,arguments",
    [
        ("@qfunc\ndef bad(x:QNum,q:QBit):\n    X(x)\n", "x,q"),
        ("@qfunc\ndef bad(x:Const[QNum],q:QBit):\n    X(x)\n", "x,q"),
        ("@qfunc\ndef bad(x:Const[QNum],q:QBit):\n    q ^= x>0.5\n", "x,x"),
        ("@qfunc\ndef bad(x:Const[QNum],q:Output[QBit]):\n    allocate(q)\n", "x,q"),
        (
            "@qfunc\ndef bad(x:Const[QNum],q:QBit):\n    a:QNum[2]=QNum()\n    allocate(a)\n    drop(a)\n",
            "x,q",
        ),
    ],
)
def test_callback_cannot_mutate_predicate_alias_or_publish_storage(
    helper: str, arguments: str
) -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "x:Output[QNum[3,False,1]],q:Output[QBit]",
                "allocate(x)\nallocate(q)\nhadamard_transform(x)\n"
                + f"control(x>0.5,lambda:bad({arguments}))",
                helper,
            )
        )


@pytest.mark.parametrize("constraint", ["qubits", "workspace", "operations", "memory"])
def test_callback_scratch_charges_actual_budgets(constraint: str) -> None:
    helper = "@qfunc\ndef payoff(x:Const[QNum],q:QBit):\n    q ^= x>1.5\n    control(x>0.5,lambda:R(0.73,0,q))\n"
    entry = definition(
        "x:Output[QNum[3,True,1]],q:Output[QBit]",
        "allocate(x)\nallocate(q)\nhadamard_transform(x)\ncontrol(x>0.5,lambda:payoff(x,q))",
        helper,
    )
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if constraint == "qubits":
            synth.synthesize(entry, max_qubits=4)
        elif constraint == "workspace":
            synth.synthesize(entry, max_workspace=0)
        elif constraint == "operations":
            synth.synthesize(entry, max_operations=1)
        else:
            synth.synthesize(entry, memory_limit_bytes=16)


@pytest.mark.parametrize("initial", [0, 1, 2, 3])
def test_callback_word_xor_keeps_arbitrary_nonzero_target(initial: int) -> None:
    helper = "@qfunc\ndef payoff(x:Const[QNum],target:QNum[2]):\n    target ^= (x>0.5)|(x<0)\n"
    body = (
        "allocate(x)\ntarget |= " + str(initial) + "\nhadamard_transform(x)\n"
        "control(x<1.5,lambda:payoff(x,target))"
    )
    program = synth.synthesize(
        definition("x:Output[QNum[3,True,1]],target:Output[QNum[2]]", body, helper)
    )
    matrix = torch.zeros((32, 32), dtype=torch.complex128)
    for raw in range(8):
        value = Fraction(raw - 8 if raw >= 4 else raw, 2)
        mask = int(value < Fraction(3, 2) and (value > Fraction(1, 2) or value < 0))
        for word in range(4):
            matrix[4 * raw + (word ^ mask), 4 * raw + word] = 1
    reference = fq.Circuit(5, dtype=torch.complex128).h(0).h(1).h(2)
    for bit in range(2):
        if initial >> bit & 1:
            reference.x(4 - bit)
    reference.any(*range(5), matrix=matrix)
    actual = fq.run(program).statevector().reshape(32, -1)
    torch.testing.assert_close(
        actual[:, 0], fq.run(reference).statevector().flatten(), atol=1e-12, rtol=0
    )
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22


def test_callback_cannot_mutate_an_inner_predicate_dependency_before_cleanup() -> None:
    helper = (
        "@qfunc\ndef bad(x:Const[QNum],q:QBit):\n"
        "    with within():\n        local:QNum[1] = x>0.5\n"
        "    with apply():\n        X(x)\n"
    )
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "x:Output[QNum[3,False,1]],q:Output[QBit]",
                "allocate(x)\nallocate(q)\ncontrol(x>0.5,lambda:bad(x,q))",
                helper,
            )
        )
