"""Bounded modular updates preserve coherent controls and complete raw words."""

import cmath
from itertools import product

import pytest
import torch

import flagquantum as fq
from benchmarks.synthesis.verification import classical_apply, encode
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("width", [1, 2, 3, 5])
@pytest.mark.parametrize("offset", [-256, -9, -1, 0, 1, 3, 256])
@pytest.mark.parametrize("helper", [False, True])
def test_every_basis_wraps_without_changing_work(
    width: int, offset: int, helper: bool
) -> None:
    helpers = "@qfunc\ndef update(word:QNum):\n    word += " + str(offset) + "\n"
    program = synth.synthesize(
        definition(
            f"x:Output[QNum[{width}]]",
            "allocate(x)\n" + ("update(x)" if helper else f"x += {offset}"),
            helpers if helper else "",
        )
    )
    wires = (program.registers["x"],)
    for value in range(1 << width):
        assert classical_apply(program.circuit, encode((value,), wires)) == encode(
            ((value + offset) % (1 << width),), wires
        )


@pytest.mark.parametrize("offset", [-3, -1, 0, 1, 3])
@pytest.mark.parametrize("condition", ["coin", "coin>=2"])
def test_controlled_helper_matches_independent_complex_permutation(
    offset: int, condition: str
) -> None:
    coin_width = 1 if condition == "coin" else 2
    width = 3 + coin_width
    helper = (
        "@qfunc\ndef update(k:int,word:QNum):\n    word += k\n"
        "@qfunc\ndef step(direction:QNum,position:QNum):\n"
        f"    control(direction{' >= 2' if coin_width == 2 else ''},lambda:update({offset},position))\n"
    )
    program = synth.synthesize(
        definition(
            f"coin:Output[QNum[{coin_width}]],x:Output[QNum[3]]",
            "allocate(coin)\nallocate(x)\nhadamard_transform(coin)\nhadamard_transform(x)\n"
            "phase(0.19*coin+0.23*x+0.31,1)\nstep(coin,x)",
            helper,
        )
    )
    permutation = torch.zeros((1 << width, 1 << width), dtype=torch.complex128)
    for coin, word in product(range(1 << coin_width), range(8)):
        active = bool(coin) if coin_width == 1 else coin >= 2
        after = (word + offset) % 8 if active else word
        permutation[coin * 8 + after, coin * 8 + word] = torch.tensor(
            cmath.exp(1j * (0.19 * coin + 0.23 * word + 0.31)),
            dtype=torch.complex128,
        )
    reference = fq.Circuit(width, dtype=torch.complex128)
    for bit in range(width):
        reference.h(bit)
    reference.any(*range(width), matrix=permutation)
    compare_states(program, reference, ("coin", "x"))


@pytest.mark.parametrize("operator", ["+=", "-="])
def test_root_offset_binding_and_inverse(operator: str) -> None:
    program = synth.synthesize(
        definition(
            "k:int,x:Output[QNum[3]]",
            "allocate(x)\nhadamard_transform(x)\nphase(0.17*x,1)\n"
            f"x {operator} k\nwith adjoint():\n    x {operator} k",
        ),
        classical_inputs={"k": -3},
    )
    reference = fq.Circuit(3, dtype=torch.complex128).h(0).h(1).h(2)
    reference.any(
        0,
        1,
        2,
        matrix=torch.diag(
            torch.tensor(
                [cmath.exp(0.17j * x) for x in range(8)], dtype=torch.complex128
            )
        ),
    )
    compare_states(program, reference, ("x",))


@pytest.mark.parametrize(
    "ports,body,helper,diagnostic",
    [
        ("x:Output[QNum[3]]", "x += 0", "", "uninitialized"),
        ("x:Output[QNum[3]]", "allocate(x)\nx += 257", "", "-256"),
        ("x:Output[QNum[3]]", "allocate(x)\nx += 1.0", "", "integer"),
        ("x:Output[QNum[3]]", "allocate(x)\nx += x", "", "integer"),
        ("x:Output[QNum[3,True,0]]", "allocate(x)\nx += 1", "", "raw unsigned"),
        ("x:Output[QArray[QBit]]", "allocate(3,x)\nx += 1", "", "raw unsigned"),
        (
            "x:Output[QNum[3]]",
            "allocate(x)\ncontrol(x>=2,lambda:step(x))",
            "@qfunc\ndef step(q:QNum):\n    q += 1\n",
            "disjoint",
        ),
        (
            "x:Output[QNum[3]],c:Output[QNum[2]]",
            "allocate(x)\nallocate(c)\ncontrol(c,lambda:step(x))",
            "@qfunc\ndef step(q:QNum):\n    q += 1\n",
            "scalar condition",
        ),
        (
            "x:Output[QArray[QBit]]",
            "allocate(4,x)\nstep(x[1:4])",
            "@qfunc\ndef step(q:QNum):\n    q += 1\n",
            "complete register",
        ),
        (
            "x:Output[QNum[3]],c:Output[QBit]",
            "allocate(x)\ncontrol(c,lambda:step(x))",
            "@qfunc\ndef step(q:QNum):\n    q += 0\n",
            "uninitialized",
        ),
        (
            "x:Output[QNum[3]],c:Output[QBit]",
            "allocate(x)\nallocate(c)\ncontrol(c,lambda:step(x))",
            "@qfunc\ndef step(q:Output[QNum[3]]):\n    q += 1\n",
            "raw inout",
        ),
    ],
)
def test_invalid_updates_refuse(
    ports: str, body: str, helper: str, diagnostic: str
) -> None:
    with pytest.raises(
        (synth.SynthesisError, HybridVerificationError), match=diagnostic
    ):
        synth.synthesize(definition(ports, body, helper))


@pytest.mark.parametrize("constraint", ["qubits", "operations", "memory"])
def test_update_actual_resource_refusals(constraint: str) -> None:
    entry = definition("x:Output[QNum[5]]", "allocate(x)\nx += 7")
    with pytest.raises((synth.SynthesisError, HybridVerificationError, PlanningError)):
        if constraint == "qubits":
            synth.synthesize(entry, max_qubits=4)
        elif constraint == "operations":
            synth.synthesize(entry, max_operations=1)
        else:
            synth.synthesize(entry, memory_limit_bytes=16)
