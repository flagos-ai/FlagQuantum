"""Indexed controls borrow only their scalar bit during modular updates."""

import cmath
from itertools import product

import pytest
import torch

import flagquantum as fq
from benchmarks.synthesis.verification import classical_apply, encode
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "condition,active",
    [
        ("word[0]", 1),
        ("word[0]==0", 0),
        ("word[0]==1", 1),
        ("word[0]!=0", 1),
        ("word[0]!=1", 0),
        ("0==word[0]", 0),
        ("1!=word[0]", 0),
        ("word[0:1]==0", 0),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("offset", [-3, 0, 2])
def test_disjoint_views_preserve_basis_coherence_and_clean_work(
    condition: str, active: int, packed: bool, offset: int, nested: bool
) -> None:
    target = "[data,word[1]]" if packed else "word[1:3]"
    helper = "@qfunc\ndef update(q:QNum):\n    q += " + str(offset) + "\n"
    operation = f"control({condition},lambda:update({target}))"
    if nested:
        helper += "@qfunc\ndef step(word:QNum,data:QNum):\n    " + operation + "\n"
        operation = "step(word,data)"
    basis = synth.synthesize(
        definition(
            "word:Output[QNum[3]],data:Output[QNum[2]]",
            "allocate(word)\nallocate(data)\n" + operation,
            helper,
        )
    )
    coherent = synth.synthesize(
        definition(
            "word:Output[QNum[3]],data:Output[QNum[2]]",
            "allocate(word)\nallocate(data)\nhadamard_transform(word)\nhadamard_transform(data)\nphase(0.19*word+0.23*data+0.31,1)\n"
            + operation,
            helper,
        )
    )
    permutation = torch.zeros((32, 32), dtype=torch.complex128)
    wires = tuple(basis.registers[name] for name in ("word", "data"))
    for word, data in product(range(8), range(4)):
        after_word, after_data = word, data
        if word & 1 == active:
            if packed:
                updated = (data + 4 * ((word >> 1) & 1) + offset) % 8
                after_word = (word & 5) | ((updated >> 2) << 1)
                after_data = updated & 3
            else:
                after_word = (word & 1) | (((word >> 1) + offset) % 4 << 1)
        assert classical_apply(basis.circuit, encode((word, data), wires)) == encode(
            (after_word, after_data), wires
        )
        permutation[after_word * 4 + after_data, word * 4 + data] = torch.tensor(
            cmath.exp(1j * (0.19 * word + 0.23 * data + 0.31)), dtype=torch.complex128
        )
    reference = fq.Circuit(5, dtype=torch.complex128)
    for bit in range(5):
        reference.h(bit)
    reference.any(*range(5), matrix=permutation)
    compare_states(coherent, reference, ("word", "data"))


@pytest.mark.parametrize(
    "condition,target",
    [
        ("word[0]", "word"),
        ("word[0]==0", "word[0:2]"),
        ("word[0]==1", "[data,word[0]]"),
        ("word>=2", "word[1:3]"),
        ("(word[0]==0)&(word[1]==1)", "[data,word[2]]"),
    ],
)
def test_overlapping_and_computed_predicate_reads_stay_refused(
    condition: str, target: str
) -> None:
    with pytest.raises(HybridVerificationError, match="disjoint"):
        synth.synthesize(
            definition(
                "word:Output[QNum[3]],data:Output[QNum[2]]",
                f"allocate(word)\nallocate(data)\ncontrol({condition},lambda:update({target}))",
                "@qfunc\ndef update(q:QNum):\n    q += 1\n",
            )
        )
