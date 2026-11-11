"""Scratch-free callback controls preserve exact integer and borrowing semantics."""

import math
from collections.abc import Callable
from itertools import product

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration


def permutation(width: int, transform: Callable[[int], int]) -> torch.Tensor:
    result = torch.zeros((1 << width, 1 << width), dtype=torch.complex128)
    for column in range(1 << width):
        result[transform(column), column] = 1
    return result


def check(program: synth.Program, reference: fq.Circuit) -> None:
    expected = reference.state().flatten()
    actual = fq.run(program).statevector().reshape(len(expected), -1)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-11, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22


COMPARISONS = (
    ("<", -1, lambda v: v < -1),
    ("<=", -4, lambda v: v <= -4),
    ("==", 0, lambda v: v == 0),
    ("!=", 3, lambda v: v != 3),
    (">", 0, lambda v: v > 0),
    (">=", 4, lambda v: v >= 4),
)


@pytest.mark.parametrize("operator,threshold,predicate", COMPARISONS)
def test_all_signed_words_and_both_nonzero_markers(
    operator: str, threshold: int, predicate: Callable[[int], bool]
) -> None:
    helper = f"@qfunc\ndef mark(v:QNum,t:QBit):\n    control(v{operator}{threshold},lambda:X(t))\n"
    for raw, marker in product(range(8), range(2)):
        program = synth.synthesize(
            definition(
                "c:Output[QBit],v:Output[QNum[3,True,0]],t:Output[QBit]",
                f"allocate(c)\nallocate(v)\nallocate(t)\nc ^= 1\nv ^= {raw}\nt ^= {marker}\n"
                "control(c,lambda:mark(v,t))",
                helper,
            ),
            max_workspace=0,
        )
        signed = raw if raw < 4 else raw - 8
        expected = torch.zeros(32, dtype=torch.complex128)
        expected[16 | (raw << 1) | (marker ^ predicate(signed))] = 1
        torch.testing.assert_close(
            fq.run(program).statevector().flatten(), expected, atol=1e-12, rtol=0
        )


@pytest.mark.parametrize("inverse", [False, True])
def test_signed_predicate_coherence_and_conditional_scalar_phase(inverse: bool) -> None:
    helper = "@qfunc\ndef mark(v:QNum,t:QBit):\n    control(v>0,lambda:X(t))\n    phase(1.0,0.23)\n"
    call = "control(c,lambda:mark(v,t))"
    if inverse:
        call = "with adjoint():\n    " + call
    program = synth.synthesize(
        definition(
            "c:Output[QBit],v:Output[QNum[3,True,0]],t:Output[QBit]",
            "allocate(c)\nallocate(v)\nallocate(t)\nH(c)\nhadamard_transform(v)\nH(t)\n"
            "RZ(0.17,c)\nphase(v,0.19)\nRZ(-0.31,t)\n" + call,
            helper,
        ),
        max_workspace=0,
    )
    reference = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(5):
        reference.h(wire)
    reference.rz(0, 0.17).rz(4, -0.31)
    values = torch.tensor([0, 1, 2, 3, -4, -3, -2, -1], dtype=torch.float64)
    reference.any(1, 2, 3, matrix=torch.diag(torch.exp(0.19j * values)))
    matrix = permutation(5, lambda b: b ^ int(bool(b & 16) and 0 < ((b >> 1) & 7) < 4))
    phase = complex(math.cos(0.23), (-1 if inverse else 1) * math.sin(0.23))
    matrix[:, 16:] *= phase
    reference.any(*range(5), matrix=matrix)
    check(program, reference)


@pytest.mark.parametrize(
    "condition,select",
    [
        ("coin==0", lambda c: c == 0),
        ("coin!=1", lambda c: c != 1),
        ("1==coin", lambda c: c == 1),
        ("coin!=0", lambda c: c != 0),
    ],
)
@pytest.mark.parametrize("delta", [-1, 1])
@pytest.mark.parametrize("inverse", [False, True])
def test_nested_scalar_updates_keep_phase_and_zero_work(
    condition: str,
    select: Callable[[int], bool],
    delta: int,
    inverse: bool,
) -> None:
    helper = f"""
@qfunc
def increment(word:QNum):
    word += {delta}
@qfunc
def step(coin:QBit,word:QNum):
    control({condition},lambda:increment(word))
    phase(1.0,0.23)
"""
    call = "control(outer,lambda:step(coin,word))"
    if inverse:
        call = "with adjoint():\n    " + call
    program = synth.synthesize(
        definition(
            "outer:Output[QBit],coin:Output[QBit],word:Output[QNum[3]]",
            "allocate(outer)\nallocate(coin)\nallocate(word)\nH(outer)\nH(coin)\nhadamard_transform(word)\n"
            "RZ(0.17,outer)\nRZ(-0.31,coin)\nRZ(0.19,word[0])\nRZ(0.38,word[1])\nRZ(0.76,word[2])\n"
            + call,
            helper,
        ),
        max_workspace=0,
    )
    reference = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(5):
        reference.h(wire)
    reference.rz(0, 0.17).rz(1, -0.31)
    for wire in range(2, 5):
        reference.rz(wire, 0.19 * (1 << (4 - wire)))

    def update(basis: int) -> int:
        active = bool(basis & 16) and select((basis >> 3) & 1)
        offset = (-delta if inverse else delta) if active else 0
        return (basis & 24) | ((basis + offset) & 7)

    matrix = permutation(5, update)
    matrix[:, 16:] *= complex(math.cos(0.23), (-1 if inverse else 1) * math.sin(0.23))
    reference.any(*range(5), matrix=matrix)
    check(program, reference)


def test_signed_metadata_uses_alias_formal_even_with_different_root_name() -> None:
    helper = "@qfunc\ndef mark(v:QNum,t:QBit):\n    control(v<0,lambda:X(t))\n"
    program = synth.synthesize(
        definition(
            "coin:Output[QBit],velocity:Output[QNum[3,True,0]],markers:Output[QNum[2]]",
            "allocate(coin)\nallocate(velocity)\nallocate(markers)\nH(coin)\nhadamard_transform(velocity)\nhadamard_transform(markers)\n"
            "RZ(0.37,markers[1])\nRZ(-0.61,markers[0])\ncontrol(coin,lambda:mark(velocity,markers[0]))",
            helper,
        ),
        max_workspace=0,
    )
    reference = fq.Circuit(6, dtype=torch.complex128)
    for wire in range(6):
        reference.h(wire)
    reference.rz(4, 0.37).rz(5, -0.61)
    reference.any(
        *range(6),
        matrix=permutation(6, lambda b: b ^ int(bool(b & 32) and bool(b & 16))),
    )
    check(program, reference)


def test_negative_const_selector_retains_checked_computation() -> None:
    helper = """
@qfunc
def increment(word:QNum):
    word += 1
@qfunc
def step(coin:Const[QBit],word:QNum):
    control(coin==0,lambda:increment(word))
"""
    entry = definition(
        "outer:Output[QBit],coin:Output[QBit],word:Output[QNum[2]]",
        "allocate(outer)\nallocate(coin)\nallocate(word)\nH(outer)\nH(coin)\nhadamard_transform(word)\nRZ(0.31,word[0])\n"
        "control(outer,lambda:step(coin,word))",
        helper,
    )
    program = synth.synthesize(entry, max_workspace=1)
    reference = fq.Circuit(4, dtype=torch.complex128)
    for wire in range(4):
        reference.h(wire)
    reference.rz(3, 0.31)
    reference.any(
        *range(4),
        matrix=permutation(
            4, lambda b: (b & 12) | ((b + int(bool(b & 8) and not bool(b & 4))) & 3)
        ),
    )
    check(program, reference)
    assert program.report["workspace_qubits"] == 1
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(entry, max_workspace=0)


@pytest.mark.parametrize(
    "body,helper,ports,initialization",
    [
        (
            "control(c,lambda:bad(v))",
            "@qfunc\ndef bad(v:QNum):\n    control(v<2,lambda:X(v[2]))\n",
            "c:Output[QBit],v:Output[QNum[3]]",
            "",
        ),
        (
            "control(c,lambda:bad(v,t))",
            "@qfunc\ndef bad(v:QNum,t:Const[QBit]):\n    control(v<2,lambda:X(t))\n",
            "c:Output[QBit],v:Output[QNum[3]],t:Output[QBit]",
            "allocate(t)\n",
        ),
        (
            "control(c,lambda:bad(v,t))",
            "@qfunc\ndef bad(v:QNum,t:QBit):\n    control(v==v,lambda:X(t))\n",
            "c:Output[QBit],v:Output[QNum[9]],t:Output[QBit]",
            "allocate(t)\n",
        ),
        (
            "control(c,lambda:bad(v,t))",
            "@qfunc\ndef bad(v:QNum,t:QBit):\n    control(v<2,lambda:X(t))\n",
            "c:Output[QBit],v:Output[QNum[3]],t:Output[QBit]",
            "",
        ),
    ],
)
def test_predicate_alias_const_width_and_uninitialized_refuse(
    body: str, helper: str, ports: str, initialization: str
) -> None:
    with pytest.raises((synth.SynthesisError, HybridVerificationError)):
        synth.synthesize(
            definition(
                ports,
                "allocate(c)\nallocate(v)\nH(c)\n" + initialization + body,
                helper,
            )
        )


def test_fixed_point_callback_keeps_numeric_fallback() -> None:
    helper = "@qfunc\ndef mark(v:QNum,t:QBit):\n    control(v<0.5,lambda:X(t))\n"
    entry = definition(
        "c:Output[QBit],v:Output[QNum[3,True,1]],t:Output[QBit]",
        "allocate(c)\nallocate(v)\nallocate(t)\nH(c)\nhadamard_transform(v)\nH(t)\n"
        "RZ(0.37,t)\ncontrol(c,lambda:mark(v,t))",
        helper,
    )
    program = synth.synthesize(entry, max_workspace=2)
    assert program.report["workspace_qubits"] == 2
    reference = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(5):
        reference.h(wire)
    reference.rz(4, 0.37)

    def update(basis: int) -> int:
        raw = (basis >> 1) & 7
        value = (raw if raw < 4 else raw - 8) / 2
        return basis ^ int(bool(basis & 16) and value < 0.5)

    reference.any(*range(5), matrix=permutation(5, update))
    check(program, reference)
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(entry, max_workspace=0)


def test_large_parent_slice_uses_existing_bounded_view_path() -> None:
    helper = "@qfunc\ndef mark(v:QNum,t:QBit):\n    control(v[1:3]==2,lambda:X(t))\n"
    program = synth.synthesize(
        definition(
            "c:Output[QBit],v:Output[QNum[12]],t:Output[QBit]",
            "allocate(c)\nallocate(v)\nallocate(t)\nH(c)\nH(v[1])\nH(v[2])\nH(t)\n"
            "v ^= 2049\nRZ(0.37,t)\ncontrol(c,lambda:mark(v,t))",
            helper,
        ),
        max_workspace=1,
    )
    assert program.report["workspace_qubits"] == 1
    expected = torch.zeros(1 << 14, dtype=torch.complex128)
    for coin, view, marker in product(range(2), range(4), range(2)):
        raw = 2049 | (view << 1)
        output = marker ^ int(bool(coin) and view == 2)
        angle = 0.37 * (marker - 0.5)
        expected[(coin << 13) | (raw << 1) | output] = torch.tensor(
            complex(math.cos(angle), math.sin(angle)) / math.sqrt(16),
            dtype=torch.complex128,
        )
    actual = fq.run(program).statevector().reshape(len(expected), -1)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-11, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22


@pytest.mark.parametrize("budget", ["width", "operations", "memory"])
def test_phase_callback_obeys_actual_budgets(budget: str) -> None:
    entry = definition(
        "c:Output[QBit],v:Output[QNum[3]],t:Output[QBit]",
        "allocate(c)\nallocate(v)\nallocate(t)\nH(c)\nhadamard_transform(v)\n"
        "control(c,lambda:mark(v,t))",
        "@qfunc\ndef mark(v:QNum,t:QBit):\n    control(v<2,lambda:X(t))\n",
    )
    with pytest.raises((synth.SynthesisError, PlanningError)):
        synth.synthesize(
            entry,
            max_workspace=0,
            max_qubits=4 if budget == "width" else 5,
            max_operations=1 if budget == "operations" else 10000,
            memory_limit_bytes=1 if budget == "memory" else 1024**2,
        )


def test_within_apply_reverses_controlled_predicate_computation() -> None:
    helper = "@qfunc\ndef mark(v:QNum,t:QBit):\n    control(v<2,lambda:X(t))\n"
    program = synth.synthesize(
        definition(
            "c:Output[QBit],v:Output[QNum[3]],t:Output[QBit]",
            "allocate(c)\nallocate(v)\nallocate(t)\nH(c)\nhadamard_transform(v)\n"
            "with within():\n    control(c,lambda:mark(v,t))\n"
            "with apply():\n    RZ(0.37,t)",
            helper,
        ),
        max_workspace=0,
    )
    reference = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(4):
        reference.h(wire)
    diagonal = torch.tensor(
        [
            (
                complex(math.cos(0.185), math.sin(0.185))
                if bool(b & 16) and ((b >> 1) & 7) < 2
                else complex(math.cos(0.185), -math.sin(0.185))
            )
            for b in range(32)
        ],
        dtype=torch.complex128,
    )
    reference.any(*range(5), matrix=torch.diag(diagonal))
    check(program, reference)


@pytest.mark.parametrize("condition", ["coin==True", "False!=coin"])
def test_scalar_comparison_does_not_coerce_bool_literals(condition: str) -> None:
    helper = f"""
@qfunc
def increment(word:QNum):
    word += 1
@qfunc
def step(coin:QBit,word:QNum):
    control({condition},lambda:increment(word))
"""
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "c:Output[QBit],coin:Output[QBit],word:Output[QNum[2]]",
                "allocate(c)\nallocate(coin)\nallocate(word)\nH(c)\ncontrol(c,lambda:step(coin,word))",
                helper,
            )
        )


def test_output_borrow_cannot_escape_controlled_callback() -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "c:Output[QBit],word:Output[QNum[3]]",
                "allocate(c)\nallocate(word)\nH(c)\ncontrol(c,lambda:create(word[1:3]))",
                "@qfunc\ndef create(word:Output[QNum[2]]):\n    allocate(word)\n",
            )
        )
