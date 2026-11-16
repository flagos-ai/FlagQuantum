"""Whole and partial mutable helper borrows preserve exact phase without scratch."""

import cmath

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_nested_borrows_keep_word_predicate_and_scalar_phase(
    partial: bool, inverse: bool
) -> None:
    helpers = """
@qfunc
def kick(word:QNum,marker:QBit):
    phase(word,0.17)
    control(word+1<=3,lambda:phase(marker,0.37))
    phase(1.0,0.23)
@qfunc
def nested(word:QNum,marker:QBit):
    kick(word,marker)
"""
    width = 3 if partial else 2
    view = "data[1:3]" if partial else "data"
    call = f"nested({view},marker)"
    if inverse:
        call = "with adjoint():\n    " + call
    program = synth.synthesize(
        definition(
            f"data:Output[QNum[{width}]],marker:Output[QBit]",
            "allocate(data)\nallocate(marker)\nhadamard_transform(data)\nH(marker)\n"
            "RZ(0.29,marker)\n" + call,
            helpers,
        ),
        max_workspace=0,
    )
    phases = []
    for raw in range(1 << width):
        word = raw >> 1 if partial else raw
        for marker in range(2):
            angle = 0.17 * word + 0.37 * marker * (word + 1 <= 3) + 0.23
            phases.append(cmath.exp(1j * (-angle if inverse else angle)))
    reference = fq.Circuit(width + 1, dtype=torch.complex128)
    for wire in range(width + 1):
        reference.h(wire)
    reference.rz(width, 0.29)
    reference.any(
        *range(width + 1),
        matrix=torch.diag(torch.tensor(phases, dtype=torch.complex128)),
    )
    assert program.report["workspace_qubits"] == 0
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        reference.state().flatten(),
        atol=1e-11,
        rtol=0,
    )


def test_preparation_helper_within_apply_has_every_nonzero_marker_column() -> None:
    helper = """
@qfunc
def prepare(word:QNum,marker:QBit):
    with within():
        H(marker)
    with apply():
        control(word+1<=3,lambda:phase(marker,3.141592653589793))
"""
    for raw in range(4):
        for marker in range(2):
            program = synth.synthesize(
                definition(
                    "word:Output[QNum[2]],marker:Output[QBit]",
                    f"allocate(word)\nallocate(marker)\nword ^= {raw}\nmarker ^= {marker}\nprepare(word,marker)",
                    helper,
                ),
                max_workspace=0,
            )
            expected = torch.zeros(8, dtype=torch.complex128)
            expected[2 * raw + (marker ^ (raw <= 2))] = 1
            torch.testing.assert_close(
                fq.run(program).statevector().flatten(), expected, atol=1e-12, rtol=0
            )


@pytest.mark.parametrize(
    "body",
    [
        "phase(word-word+1,0.37)",
        "control((word==0) ^ (word==0),lambda:phase(1.0,0.37))",
    ],
)
def test_borrowed_phase_cancellation_does_not_erase_uninitialized_reads(
    body: str,
) -> None:
    with pytest.raises(HybridVerificationError, match="initializ"):
        synth.synthesize(
            definition(
                "word:Output[QNum[2]]",
                "kick(word)",
                f"@qfunc\ndef kick(word:QNum):\n    {body}\n",
            ),
        )


def test_borrowed_fixed_point_comparison_retains_checked_numeric_fallback() -> None:
    helper = "@qfunc\ndef kick(word:QNum,marker:QBit):\n    control(word<0.5,lambda:phase(marker,0.37))\n"
    entry = definition(
        "word:Output[QNum[3,True,1]],marker:Output[QBit]",
        "allocate(word)\nallocate(marker)\nhadamard_transform(word)\nH(marker)\nkick(word,marker)",
        helper,
    )
    program = synth.synthesize(entry)
    expected = torch.tensor(
        [
            cmath.exp(0.37j * marker * ((raw if raw < 4 else raw - 8) / 2 < 0.5)) / 4
            for raw in range(8)
            for marker in range(2)
        ],
        dtype=torch.complex128,
    )
    actual = fq.run(program).statevector().reshape(16, -1)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22
    assert program.report["workspace_qubits"] != 0
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(entry, max_workspace=0)


@pytest.mark.parametrize("width", [7, 8])
@pytest.mark.parametrize("delta", [-1, 1])
def test_wide_callback_updates_keep_carry_phase_and_computed_fallback(
    width: int, delta: int
) -> None:
    rotations = "\n".join(
        f"RZ({0.073 * (1 << bit)},word[{bit}])" for bit in range(width)
    )
    entry = definition(
        f"coin:Output[QBit],word:Output[QNum[{width}]]",
        "allocate(coin)\nallocate(word)\nH(coin)\nhadamard_transform(word)\n"
        "RZ(0.31,coin)\n" + rotations + "\ncontrol(coin,lambda:step(word))",
        f"@qfunc\ndef advance(word:QNum):\n    word += {delta}\n"
        "@qfunc\ndef step(word:QNum):\n    advance(word)\n",
    )
    program = synth.synthesize(entry)
    size = 1 << (width + 1)
    expected = torch.empty(size, dtype=torch.complex128)
    for coin in range(2):
        for raw in range(1 << width):
            angle = 0.31 * (coin - 0.5) + 0.073 * (raw - ((1 << width) - 1) / 2)
            output = (coin << width) | ((raw + coin * delta) % (1 << width))
            expected[output] = torch.tensor(
                cmath.exp(1j * angle) / size**0.5, dtype=torch.complex128
            )
    actual = fq.run(program).statevector().reshape(size, -1)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-11, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22
    constrained = synth.synthesize(entry, max_workspace=0)
    assert constrained.report["workspace_qubits"] == 0
    from tests.synthesis_workflow_helpers import check_program_equivalence

    check_program_equivalence(program, constrained)
