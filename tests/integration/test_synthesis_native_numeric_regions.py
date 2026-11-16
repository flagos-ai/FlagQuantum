"""Exact lexical arithmetic scratch in root and borrowed unitary preparation."""

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


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize(
    "expression,inputs,local",
    [
        ("max(x*1.0,x*0.75+y*0.75-1.25)", "QNum[2]", "QNum[5,False,2]"),
        ("x-y", "QNum[2,True,1]", "QNum[5,True,1]"),
        ("-x", "QNum[2,True,1]", "QNum[3,True,1]"),
    ],
)
def test_exact_local_and_inverse_keep_independent_full_state(
    borrowed: bool, inverse: bool, expression: str, inputs: str, local: str
) -> None:
    region = f"with within():\n    score:{local} = {expression}\nwith apply():\n    control(score>0.5,lambda:R(0.73,1.5707963267948966,q))"
    helper = ""
    if borrowed:
        helper = (
            f"@qfunc\ndef prepare(x:{inputs},y:{inputs},q:QBit):\n"
            + "\n".join("    " + line for line in region.splitlines())
            + "\n"
        )
        region = "prepare(x,y,q)"
    if inverse:
        region = "with adjoint():\n" + "\n".join(
            "    " + line for line in region.splitlines()
        )
    program = synth.synthesize(
        definition(
            f"x:Output[{inputs}],y:Output[{inputs}],q:Output[QBit]",
            "allocate(x)\nallocate(y)\nallocate(q)\nhadamard_transform(x)\nhadamard_transform(y)\nX(q)\nH(q)\nS(q)\n"
            + region,
            helper,
        ),
        memory_limit_bytes=2**27,
    )
    matrix = torch.eye(32, dtype=torch.complex128)
    theta = -0.73 if inverse else 0.73
    rotation = torch.tensor(
        [
            [math.cos(theta / 2), -math.sin(theta / 2)],
            [math.sin(theta / 2), math.cos(theta / 2)],
        ],
        dtype=torch.complex128,
    )
    for a in range(4):
        for b in range(4):
            x = Fraction(a - 4 if a >= 2 else a, 2) if "True" in inputs else Fraction(a)
            y = Fraction(b - 4 if b >= 2 else b, 2) if "True" in inputs else Fraction(b)
            value = (
                max(x, Fraction(3 * (a + b) - 5, 4))
                if expression.startswith("max")
                else x - y if expression == "x-y" else -x
            )
            if value > Fraction(1, 2):
                start = 2 * (4 * a + b)
                matrix[start : start + 2, start : start + 2] = rotation
    reference = (
        fq.Circuit(5, dtype=torch.complex128)
        .h(0)
        .h(1)
        .h(2)
        .h(3)
        .x(4)
        .h(4)
        .s(4)
        .any(*range(5), matrix=matrix)
    )
    actual = fq.run(program).statevector().reshape(32, -1)
    torch.testing.assert_close(
        actual[:, 0], fq.run(reference).statevector().flatten(), atol=1e-12, rtol=0
    )
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22


@pytest.mark.parametrize(
    "region",
    [
        "with within():\n    score:QNum[5] = x*0.5\nwith apply():\n    phase(score,0.73)",
        "with within():\n    score:QNum[1,False,1] = x*0.5\nwith apply():\n    phase(score,0.73)",
        "with within():\n    score:QNum[5,False,1] = -x\nwith apply():\n    phase(score,0.73)",
        "with within():\n    score:QNum[4,True,0] = -x\nwith apply():\n    control(score<0,lambda:X(x))",
        "with within():\n    score:QNum[4,True,0] = -x\nwith apply():\n    phase(score,0.73)\nphase(score,0.5)",
    ],
)
def test_numeric_local_rejects_truncation_format_mutated_dependency_and_escape(
    region: str,
) -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "x:Output[QNum[2]],q:Output[QBit]",
                "allocate(x)\nallocate(q)\nhadamard_transform(x)\n" + region,
            )
        )


@pytest.mark.parametrize("constraint", ["qubits", "workspace", "operations", "memory"])
def test_numeric_local_work_and_operations_charge_existing_budgets(
    constraint: str,
) -> None:
    entry = definition(
        "x:Output[QNum[2]],q:Output[QBit]",
        "allocate(x)\nallocate(q)\nhadamard_transform(x)\nwith within():\n    score:QNum[3,True,0] = -x\nwith apply():\n    control(score<0,lambda:R(0.73,0,q))",
    )
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if constraint == "qubits":
            synth.synthesize(entry, max_qubits=3)
        elif constraint == "workspace":
            synth.synthesize(entry, max_workspace=0)
        elif constraint == "operations":
            synth.synthesize(entry, max_operations=1)
        else:
            synth.synthesize(entry, memory_limit_bytes=16)
