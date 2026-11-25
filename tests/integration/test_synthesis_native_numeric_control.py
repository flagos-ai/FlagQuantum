"""Exact numeric controls preserve coherent boundary values and branch phase."""

import cmath
import math
from fractions import Fraction
from itertools import product
from operator import eq, ge, gt, le, lt, ne

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_exact_numbers import definition
from tests.synthesis_workflow_helpers import check_state

pytestmark = pytest.mark.integration
RELATIONS = {"<": lt, "<=": le, ">": gt, ">=": ge, "==": eq, "!=": ne}


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("relation", RELATIONS)
@pytest.mark.parametrize("threshold", [-0.5, 0.5, 1.5])
@pytest.mark.parametrize("action", ["rotation", "phase"])
def test_complete_numeric_domain_against_independent_controlled_circuit(
    signed: bool, relation: str, threshold: float, action: str
) -> None:
    callback = (
        "R(0.73,1.5707963267948966,q)" if action == "rotation" else "phase(1,0.73)"
    )
    entry = definition(
        f"x:Output[QNum[3,{signed},1]],q:Output[QBit]",
        "allocate(x)\nallocate(q)\nhadamard_transform(x)\nH(q)\nphase(1,0.37)\n"
        + f"control(x{relation}{threshold},lambda:{callback})",
    )
    program = synth.synthesize(entry, max_qubits=24, memory_limit_bytes=2**24)
    matrix = torch.eye(16, dtype=torch.complex128)
    for raw in range(8):
        value = Fraction(raw - 8 if signed and raw >= 4 else raw, 2)
        if RELATIONS[relation](value, Fraction(threshold)):
            block = (
                torch.tensor(
                    [
                        [math.cos(0.73 / 2), -math.sin(0.73 / 2)],
                        [math.sin(0.73 / 2), math.cos(0.73 / 2)],
                    ],
                    dtype=torch.complex128,
                )
                if action == "rotation"
                else torch.eye(2, dtype=torch.complex128) * cmath.exp(0.73j)
            )
            matrix[raw * 2 : raw * 2 + 2, raw * 2 : raw * 2 + 2] = block
    reference = fq.Circuit(4, dtype=torch.complex128).h(0).h(1).h(2).h(3)
    reference.any(0, 1, 2, 3, matrix=matrix * cmath.exp(0.37j))
    actual = fq.run(program).statevector().reshape(16, -1)
    torch.testing.assert_close(
        actual[:, 0], fq.run(reference).statevector().flatten(), atol=1e-12, rtol=0
    )
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22
    assert tuple(program.registers) == ("x", "q")


@pytest.mark.parametrize("condition", ["(x>0.5)&(x<1.5)", "~(x>0.5)"])
@pytest.mark.parametrize("inverse", [False, True])
def test_compound_predicate_and_adjoint_preserve_conditional_global_phase(
    condition: str, inverse: bool
) -> None:
    operation = f"control({condition},lambda:phase(1,0.73))"
    if inverse:
        operation = "with adjoint():\n    " + operation
    entry = definition(
        "x:Output[QNum[3,True,1]],q:Output[QBit]",
        "allocate(x)\nallocate(q)\nhadamard_transform(x)\nH(q)\n" + operation,
    )
    program = synth.synthesize(entry)
    expected = {}
    for raw, q in product(range(8), range(2)):
        value = Fraction(raw - 8 if raw >= 4 else raw, 2)
        selected = (
            Fraction(1, 2) < value < Fraction(3, 2)
            if "&" in condition
            else not value > Fraction(1, 2)
        )
        expected[(raw, q)] = (
            cmath.exp((-0.73j if inverse else 0.73j) if selected else 0j) / 4
        )
    check_state(program, fq.run(program), expected)


@pytest.mark.parametrize("initial", [0, 1])
def test_two_formats_compare_values_instead_of_raw_encodings(initial: int) -> None:
    program = synth.synthesize(
        definition(
            "x:Output[QNum[2,True,1]],y:Output[QNum[2]],q:Output[QBit]",
            "allocate(x)\nallocate(y)\nallocate(q)\nhadamard_transform(x)\n"
            "hadamard_transform(y)\n"
            + ("X(q)\n" if initial else "")
            + "control(x<y,lambda:X(q))",
        )
    )
    expected = {
        (x, y, initial ^ int(Fraction(x - 4 if x >= 2 else x, 2) < y)): 0.25 + 0j
        for x, y in product(range(4), repeat=2)
    }
    check_state(program, fq.run(program), expected)


@pytest.mark.parametrize(
    "body,ports,diagnostic",
    [
        (
            "allocate(x)\nallocate(q)\ncontrol(x,lambda:R(0.7,0,q))",
            "x:Output[QNum[3,True,1]],q:Output[QBit]",
            "numeric reads",
        ),
        (
            "allocate(x)\nallocate(q)\ncontrol(x>0.1,lambda:R(0.7,0,q))",
            "x:Output[QNum[3,True,1]],q:Output[QBit]",
            "binary representable",
        ),
        (
            "allocate(q)\ncontrol(x>0.5,lambda:R(0.7,0,q))",
            "x:Output[QNum[3,True,1]],q:Output[QBit]",
            "uninitialized",
        ),
        (
            "allocate(x)\nallocate(q)\ncontrol(x>0.5,lambda:phase(x,0.7))",
            "x:Output[QNum[3,True,1]],q:Output[QBit]",
            "coherent callback",
        ),
        (
            "allocate(x)\nallocate(q)\ncontrol(x>0.5,lambda:R(0.7,0,q))",
            "x:Output[QNum[9,True,1]],q:Output[QBit]",
            "eight input bits",
        ),
        (
            "allocate(q)\ns:QNum[3,True,1]=QNum()\nallocate(s)\ndrop(s)\n"
            "control(s>0.5,lambda:R(0.7,0,q))",
            "q:Output[QBit]",
            "dropped",
        ),
    ],
)
def test_numeric_controls_refuse_unproved_reads_and_unsupported_callbacks(
    body: str, ports: str, diagnostic: str
) -> None:
    with pytest.raises(HybridVerificationError, match=diagnostic):
        synth.synthesize(definition(ports, body))


@pytest.mark.parametrize("constraint", ["workspace", "qubits", "operations", "memory"])
def test_control_charges_predicate_storage_and_execution_budget(
    constraint: str,
) -> None:
    entry = definition(
        "x:Output[QNum[3,True,1]],q:Output[QBit]",
        "allocate(x)\nallocate(q)\nhadamard_transform(x)\n"
        "control(x>0.5,lambda:R(0.73,1.5707963267948966,q))",
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


def test_numeric_condition_cannot_mutate_a_const_callback_target() -> None:
    helper = (
        "@qfunc\ndef forbidden(x:Const[QNum[3,True,1]],q:Const[QBit]):\n"
        "    control(x>0.5,lambda:R(0.73,0,q))\n"
    )
    with pytest.raises(HybridVerificationError, match="writable"):
        synth.synthesize(
            definition(
                "x:Output[QNum[3,True,1]],q:Output[QBit]",
                "allocate(x)\nallocate(q)\nforbidden(x,q)",
                helper,
            )
        )


def test_numeric_cancellation_cannot_erase_an_uninitialized_read() -> None:
    with pytest.raises(HybridVerificationError, match="uninitialized"):
        synth.synthesize(
            definition(
                "x:Output[QNum[3,True,1]],q:Output[QBit]",
                "allocate(q)\ncontrol(x-x==0,lambda:phase(1,0.73))",
            )
        )
