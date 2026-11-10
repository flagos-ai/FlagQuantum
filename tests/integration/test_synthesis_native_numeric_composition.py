"""Native numeric composition preserves values, phases and owned environment."""

from __future__ import annotations

import cmath
from collections.abc import Callable
from fractions import Fraction
from itertools import product
from typing import Annotated

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.synth import (
    Const,
    Output,
    QNum,
    allocate,
    drop,
    hadamard_transform,
    phase,
    qfunc,
)
from tests.integration.test_synthesis_native_exact_numbers import definition
from tests.integration.test_synthesis_register_workflow import check_state

pytestmark = pytest.mark.integration


@qfunc
def negate(value: Const[QNum], result: Output[QNum]) -> None:
    result |= -value


@qfunc
def twice(value: Const[QNum], result: Output[QNum]) -> None:
    result |= value * 2


@qfunc
def helpers(
    a: Output[Annotated[QNum, (1, True, 1)]],
    b: Output[Annotated[QNum, (1, False, 1)]],
    c: Output[Annotated[QNum, (1, False, 0)]],
    u: Output[QNum],
    v: Output[QNum],
    w: Output[QNum],
    z: Output[QNum],
) -> None:
    allocate(a)
    allocate(b)
    allocate(c)
    hadamard_transform(a)
    hadamard_transform(b)
    hadamard_transform(c)
    phase(a + b + c + 0.125, 0.37)
    negate(a, u)
    negate(b, v)
    negate(c, w)
    twice(u, z)


def test_same_width_different_sign_and_scale_specializations() -> None:
    program = synth.synthesize(helpers, max_qubits=24, memory_limit_bytes=512 * 1024**2)
    assert program.registers == {
        "a": (0,),
        "b": (1,),
        "c": (2,),
        "u": (3,),
        "v": (4,),
        "w": (5,),
        "z": (6, 7),
    }
    reference = fq.Circuit(8, dtype=torch.complex128).h(0).h(1).h(2)
    diagonal = [
        cmath.exp(0.37j * (-a / 2 + b / 2 + c + 0.125))
        for a, b, c in product(range(2), repeat=3)
    ]
    reference.any(
        0, 1, 2, matrix=torch.diag(torch.tensor(diagonal, dtype=torch.complex128))
    )
    reference.cx(0, 3).cx(1, 4).cx(2, 5).cx(0, 6)
    compare_states(program, reference, tuple(program.registers))
    decoded = program.numeric_probabilities(
        fq.run(program), registers=tuple(program.registers)
    )
    expected = {
        (
            Fraction(-a, 2),
            Fraction(b, 2),
            Fraction(c),
            Fraction(a, 2),
            Fraction(-b, 2),
            Fraction(-c),
            Fraction(a),
        ): 1
        / 8
        for a, b, c in product(range(2), repeat=3)
    }
    assert dict(decoded) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "expression,oracle",
    [
        ("a+b", lambda a, b: a + b),
        ("a-b", lambda a, b: a - b),
        ("a*b", lambda a, b: a * b),
        ("a**2", lambda a, b: a * a),
        ("min(a,b)", min),
        ("max(a,b)", max),
        ("a<b", lambda a, b: int(a < b)),
        ("a==b", lambda a, b: int(a == b)),
    ],
)
def test_signed_fractional_coherent_arithmetic(
    expression: str, oracle: Callable[[Fraction, Fraction], Fraction | int]
) -> None:
    entry = definition(
        "a:Output[QNum[1,True,1]],b:Output[QNum[1]],result:Output[QNum]",
        "allocate(a)\nallocate(b)\nhadamard_transform(a)\nhadamard_transform(b)\n"
        f"phase(a-b+0.125,0.41)\nresult |= {expression}",
    )
    program = synth.synthesize(entry, max_qubits=24, memory_limit_bytes=512 * 1024**2)
    formats = {
        name: (signed, scale)
        for name, signed, scale in program._checked.numeric_formats
    }
    signed, scale = formats["result"]
    width = len(program.registers["result"])
    expected = {}
    for ra, rb in product(range(2), repeat=2):
        a, b = Fraction(-ra, 2), Fraction(rb)
        raw = int(oracle(a, b) * (1 << scale)) & ((1 << width) - 1)
        expected[(ra, rb, raw)] = cmath.exp(0.41j * (float(a - b) + 0.125)) / 2
    check_state(program, fq.run(program), expected)
    labels = program.numeric_probabilities(
        fq.run(program),
        registers=("result",),
        max_amplitudes=1 << program.circuit.n_wires,
    )
    assert set(labels) == {
        (Fraction(oracle(Fraction(-a, 2), Fraction(b))),)
        for a, b in product(range(2), repeat=2)
    }


@qfunc
def retained_local(
    x: Output[Annotated[QNum, (1, False, 0)]], result: Output[QNum]
) -> None:
    allocate(x)
    hadamard_transform(x)
    phase(x, 0.37)
    intermediate: Annotated[QNum, (1, False, 1)] = QNum()
    intermediate |= 0.5 * x
    result |= intermediate * 2
    drop(intermediate)


def test_local_drop_preserves_entanglement_and_clean_scratch() -> None:
    program = synth.synthesize(
        retained_local, max_qubits=20, memory_limit_bytes=512 * 1024**2
    )
    state = fq.run(program).statevector().flatten()
    # Native declarations map published x/result first, then the retained local;
    # any following physical bits are genuine arithmetic scratch.
    reference = fq.Circuit(4, dtype=torch.complex128).h(0)
    reference.any(
        0,
        matrix=torch.diag(torch.tensor([1, cmath.exp(0.37j)], dtype=torch.complex128)),
    )
    reference.cx(0, 3).cx(0, 1)
    logical_width = 4
    actual = state.reshape(1 << logical_width, -1)
    expected = fq.run(reference).statevector().flatten()
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-24
    reduced = actual.reshape(8, 2, -1)
    density = torch.einsum("aew,bew->ab", reduced, reduced.conj())
    expected_density = torch.zeros((8, 8), dtype=torch.complex128)
    expected_density[0, 0] = expected_density[6, 6] = 0.5
    torch.testing.assert_close(density, expected_density, atol=1e-12, rtol=0)
    assert program.report["workspace_qubits"] == program.circuit.n_wires - logical_width


@pytest.mark.parametrize(
    "ports,body,helper,diagnostic",
    [
        (
            "x:Output[QNum[1]],y:Output[QNum]",
            "a:QNum[2,True,1]=QNum()\na |= -0.5\na |= -0.5\ny |= a\nallocate(x)\ndrop(a)",
            "",
            "initialized",
        ),
        (
            "x:Output[QNum[1]],y:Output[QNum]",
            "a:QNum[2,True,1]=QNum()\ny |= a\nallocate(x)\ndrop(a)",
            "",
            "uninitialized",
        ),
        (
            "x:Output[QNum[1]],y:Output[QNum]",
            "a:QNum[2,True,1]=QNum()\na |= -0.5\ndrop(a)\ny |= a\nallocate(x)",
            "",
            "dropped",
        ),
        (
            "x:Output[QNum[1]],y:Output[QNum]",
            "allocate(x)\ny |= 0.1*x",
            "",
            "exactly binary",
        ),
        (
            "x:Output[QNum[2,True,1]],y:Output[QNum]",
            "allocate(x)\nhelper(x,y)",
            "@qfunc\ndef helper(x:Const[QNum[2,False,1]],y:Output[QNum]):\n    y |= -x\n",
            "numeric formats",
        ),
        (
            "x:Output[QNum[2,True,1]],y:Output[QNum]",
            "allocate(x)\nhelper(x,y)",
            "@qfunc\ndef helper(x:Const[QNum[2,True,0]],y:Output[QNum]):\n    y |= -x\n",
            "numeric formats",
        ),
        (
            "x:Output[QNum[1]],y:Output[QNum[1]]",
            "allocate(x)\nhelper(x,y)",
            "@qfunc\ndef helper(x:Const[QNum],y:Output[QNum]):\n    y |= -x\n",
            "output formats",
        ),
        (
            "x:Output[QNum[1,False,1]],y:Output[QNum]",
            "helper(x,y)",
            "@qfunc\ndef helper(x:Const[QNum],y:Output[QNum]):\n    y |= -x\n",
            "uninitialized",
        ),
        (
            "x:Output[QNum[1,False,1]]",
            "allocate(x)\nhelper(x,x)",
            "@qfunc\ndef helper(x:Const[QNum],y:Output[QNum]):\n    y |= x\n",
            "disjoint",
        ),
        (
            "x:Output[QNum[1]],y:Output[QNum]",
            "allocate(x)\na:QNum[2,True,0]=QNum()\nhelper(x,a)\ny |= 0\ndrop(a)",
            "@qfunc\ndef helper(x:Const[QNum],y:Output[QNum]):\n    y |= -x\n",
            "escape",
        ),
        (
            "x:Output[QNum[1]],y:Output[QNum]",
            "allocate(x)\na:QNum[2,False,1]=QNum()\na |= 0.25\ny |= a\ndrop(a)",
            "",
            "fractional bits",
        ),
        (
            "x:Output[QNum[1]],y:Output[QNum]",
            "allocate(x)\na:QNum[2,True,0]=QNum()\na |= 2\ny |= a\ndrop(a)",
            "",
            "initial value",
        ),
        (
            "x:Output[QNum[256,True,1]],y:Output[QNum]",
            "allocate(x)\ny |= x*2",
            "",
            "256-bit",
        ),
        (
            "x:Output[QNum[1,False,1]],y:Output[QNum]",
            "allocate(x)\ncontrol(x,lambda:helper(x,y))",
            "@qfunc\ndef helper(x:Const[QNum],y:Output[QNum]):\n    y |= -x\n",
            "numeric",
        ),
    ],
)
def test_numeric_ownership_format_and_precision_refusals(
    ports: str, body: str, helper: str, diagnostic: str
) -> None:
    with pytest.raises(
        (HybridVerificationError, synth.SynthesisError), match=diagnostic
    ):
        synth.synthesize(definition(ports, body, helper))


@pytest.mark.parametrize("constraint", ["operations", "qubits", "workspace", "memory"])
def test_numeric_composition_actual_budgets(constraint: str) -> None:
    from flagquantum.errors import PlanningError

    with pytest.raises((HybridVerificationError, synth.SynthesisError, PlanningError)):
        if constraint == "operations":
            synth.synthesize(helpers, max_operations=1)
        elif constraint == "qubits":
            synth.synthesize(helpers, max_qubits=7)
        elif constraint == "workspace":
            synth.synthesize(helpers, max_workspace=0)
        else:
            synth.synthesize(helpers, memory_limit_bytes=16)


@pytest.mark.parametrize("expression", ["-0.125*x", "-(0.125*x)"])
def test_small_fractional_difference_sign_extends_storage(expression: str) -> None:
    program = synth.synthesize(
        definition(
            "x:Output[QNum[1]],result:Output[QNum]",
            f"allocate(x)\nhadamard_transform(x)\nresult |= {expression}",
        )
    )
    check_state(program, fq.run(program), {(0, 0): 2**-0.5, (1, 7): 2**-0.5})
    assert dict(program.numeric_probabilities(fq.run(program))) == pytest.approx(
        {
            (Fraction(0),): 0.5,
            (Fraction(-1, 8),): 0.5,
        }
    )


def test_explicit_numeric_helper_formats() -> None:
    helper = "@qfunc\ndef helper(x:Const[QNum[1,True,1]],y:Output[QNum[1,False,1]]):\n    y |= -x\n"
    entry = definition(
        "x:Output[QNum[1,True,1]],result:Output[QNum[1,False,1]]",
        "allocate(x)\nhadamard_transform(x)\nphase(x+0.125,0.37)\nhelper(x,result)",
        helper,
    )
    program = synth.synthesize(entry)
    reference = fq.Circuit(2, dtype=torch.complex128).h(0)
    reference.any(
        0,
        matrix=torch.diag(
            torch.tensor(
                [cmath.exp(0.37j * 0.125), cmath.exp(0.37j * -0.375)],
                dtype=torch.complex128,
            )
        ),
    ).cx(0, 1)
    compare_states(program, reference, ("x", "result"))
