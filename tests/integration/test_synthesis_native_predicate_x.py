"""Bounded predicate-X preserves XOR, coherence, scratch and borrowed roles."""

import cmath
from collections.abc import Callable

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration
PREDICATES: tuple[tuple[str, Callable[[int], bool]], ...] = (
    ("c<=0", lambda c: c <= 0),
    ("(c>=1)&(c<=2)", lambda c: 1 <= c <= 2),
    ("~(c>=2)", lambda c: c < 2),
    ("(c<=0)|(c>=3)", lambda c: c in (0, 3)),
    ("(c>=1)^(c>=2)", lambda c: c == 1),
)


def check(program: synth.Program, expected: torch.Tensor) -> None:
    work = program.report["workspace_qubits"]
    assert isinstance(work, int)
    complete = torch.zeros((8, 1 << work), dtype=torch.complex128)
    complete[:, 0] = expected
    actual = fq.run(program).statevector().flatten()
    torch.testing.assert_close(actual, complete.flatten(), atol=1e-12, rtol=0)


@pytest.mark.parametrize("condition,predicate", PREDICATES)
@pytest.mark.parametrize("word", range(4))
@pytest.mark.parametrize("marker", [0, 1])
def test_every_predicate_branch_and_nonzero_target(
    condition: str, predicate: Callable[[int], bool], word: int, marker: int
) -> None:
    program = synth.synthesize(
        definition(
            "c:Output[QNum[2]],target:Output[QBit]",
            f"c |= {word}\nallocate(target)\n"
            + ("X(target)\n" if marker else "")
            + f"control({condition},lambda:X(target))",
        )
    )
    expected = torch.zeros(8, dtype=torch.complex128)
    expected[2 * word + (marker ^ predicate(word))] = 1
    check(program, expected)


@pytest.mark.parametrize("helper", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_complex_coherence_and_borrowed_helper(helper: bool, inverse: bool) -> None:
    extra = "@qfunc\ndef shift(c:QNum,target:QBit):\n    control((c>=1)&(c<=2),lambda:X(target))\n"
    call = "shift(c,target)" if helper else "control((c>=1)&(c<=2),lambda:X(target))"
    body = (
        "allocate(c)\nallocate(target)\nhadamard_transform(c)\nH(target)\nphase(c,0.37)\nphase(target,0.29)\n"
        + ("with adjoint():\n    " + call if inverse else call)
    )
    program = synth.synthesize(
        definition("c:Output[QNum[2]],target:Output[QBit]", body, extra)
    )
    values = [0j] * 8
    for c in range(4):
        for target in range(2):
            values[2 * c + (target ^ (1 <= c <= 2))] = cmath.exp(
                1j * (0.37 * c + 0.29 * target)
            ) / (8**0.5)
    check(program, torch.tensor(values, dtype=torch.complex128))


def test_within_inverse_cleans_scratch_and_preserves_phase() -> None:
    body = "allocate(c)\nallocate(target)\nhadamard_transform(c)\nH(target)\nwith within():\n    control((c>=1)&(c<=2),lambda:X(target))\nwith apply():\n    phase(target,0.29)\n    phase(1,0.17)"
    program = synth.synthesize(
        definition("c:Output[QNum[2]],target:Output[QBit]", body)
    )
    expected = torch.tensor(
        [
            cmath.exp(1j * (0.29 * (target ^ (1 <= c <= 2)) + 0.17)) / (8**0.5)
            for c in range(4)
            for target in range(2)
        ],
        dtype=torch.complex128,
    )
    check(program, expected)


@pytest.mark.parametrize(
    "ports,body,extra",
    [
        (
            "c:Output[QNum[2]],target:Output[QBit]",
            "allocate(target)\ncontrol(c>=1,lambda:X(target))\nallocate(c)",
            "",
        ),
        ("c:Output[QBit]", "allocate(c)\ncontrol(c>=1,lambda:X(c))", ""),
        (
            "c:Output[QNum[9]],target:Output[QBit]",
            "allocate(c)\nallocate(target)\ncontrol(c>=1,lambda:X(target))",
            "",
        ),
        (
            "c:Output[QNum[2]],target:Output[QBit]",
            "allocate(c)\nallocate(target)\nbad(c,target)",
            "@qfunc\ndef bad(c:QNum,target:Const[QBit]):\n    control(c>=1,lambda:X(target))\n",
        ),
        (
            "c:Output[QNum[2]],target:Output[QBit]",
            "allocate(c)\nallocate(target)\nwith adjoint():\n    publish(c,target)",
            "@qfunc\ndef publish(c:QNum,target:Output[QBit]):\n    allocate(target)\n",
        ),
    ],
)
def test_alias_initialization_width_const_and_output_refusal(
    ports: str, body: str, extra: str
) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(definition(ports, body, extra))


def test_partial_borrowed_scalar_predicate_reads_only_its_selected_bit() -> None:
    extra = (
        "@qfunc\ndef shift(c:QBit,target:QBit):\n    control(c>=1,lambda:X(target))\n"
    )
    entry = definition(
        "c:Output[QArray[QBit]],target:Output[QBit]",
        "allocate(2,c)\nallocate(target)\nhadamard_transform(c)\nshift(c[0],target)",
        extra,
    )
    program = synth.synthesize(entry)
    reference = fq.Circuit(3, dtype=torch.complex128).h(0).h(1).cx(1, 2)
    actual = fq.run(program).statevector().reshape(8, -1)
    torch.testing.assert_close(
        actual[:, 0], reference.state().flatten(), atol=1e-12, rtol=0
    )
    assert torch.count_nonzero(actual[:, 1:]) == 0
