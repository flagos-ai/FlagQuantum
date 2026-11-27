"""Checked quantum addends preserve source values, coherence and modular targets."""

import cmath
from itertools import product

import pytest
import torch

import flagquantum as fq
from benchmarks.synthesis.verification import classical_apply, encode
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler.frontends._structured.verifier import HybridVerificationError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("source_width,target_width", ((1, 1), (2, 3), (3, 2), (3, 3)))
@pytest.mark.parametrize("sign", (1, -1))
@pytest.mark.parametrize("helper", (False, True))
def test_every_basis_pair_and_absolute_coherent_state(
    source_width: int, target_width: int, sign: int, helper: bool
) -> None:
    operator = "+=" if sign == 1 else "-="
    helpers = (
        f"@qfunc\ndef update(left:Const[QNum],right:QNum):\n    right {operator} left\n"
        if helper
        else ""
    )
    action = "update(left,right)" if helper else f"right {operator} left"
    ports = f"left:Output[QNum[{source_width}]],right:Output[QNum[{target_width}]]"
    preparation = "allocate(left)\nallocate(right)\n"
    basis = synth.synthesize(definition(ports, preparation + action, helpers))
    coherent = synth.synthesize(
        definition(
            ports,
            preparation
            + "hadamard_transform(left)\nhadamard_transform(right)\n"
            + "phase(0.19*left+0.23*right+0.31,1)\n"
            + action,
            helpers,
        )
    )
    width = source_width + target_width
    permutation = torch.zeros((1 << width, 1 << width), dtype=torch.complex128)
    wires = tuple(basis.registers[n] for n in ("left", "right"))
    for left, right in product(range(1 << source_width), range(1 << target_width)):
        updated = (right + sign * left) % (1 << target_width)
        assert classical_apply(basis.circuit, encode((left, right), wires)) == encode(
            (left, updated), wires
        )
        permutation[
            (left << target_width) | updated, (left << target_width) | right
        ] = torch.tensor(
            cmath.exp(1j * (0.19 * left + 0.23 * right + 0.31)), dtype=torch.complex128
        )
    reference = fq.Circuit(width, dtype=torch.complex128)
    for bit in range(width):
        reference.h(bit)
    reference.any(*range(width), matrix=permutation)
    compare_states(coherent, reference, ("left", "right"))


def test_original_accumulator_initialized_to_four() -> None:
    program = synth.synthesize(
        definition(
            "y:Output[QNum],acc:Output[QNum]",
            "allocate(2,y)\nhadamard_transform(y)\nacc |= 4\nacc += y",
        )
    )
    result = fq.run(program)
    assert dict(program.probabilities(result, registers=("y", "acc"))) == pytest.approx(
        {(y, 4 + y): 0.25 for y in range(4)}
    )


def test_disjoint_slices_of_one_register_keep_logical_low_bit_order() -> None:
    program = synth.synthesize(
        definition(
            "q:Output[QNum[5]]",
            "allocate(q)\nupdate(q[0:2],q[2:5])",
            "@qfunc\ndef update(left:QNum,right:QNum):\n    right += left\n",
        )
    )
    wire_map = (program.registers["q"],)
    for word in range(32):
        expected = (word & 3) | ((((word >> 2) + (word & 3)) % 8) << 2)
        assert classical_apply(program.circuit, encode((word,), wire_map)) == encode(
            (expected,), wire_map
        )


@pytest.mark.parametrize("action", ("right += left", "right -= left"))
def test_adjoint_round_trip_preserves_arbitrary_relative_and_global_phase(
    action: str,
) -> None:
    program = synth.synthesize(
        definition(
            "left:Output[QNum[2]],right:Output[QNum[3]]",
            "allocate(left)\nallocate(right)\nhadamard_transform(left)\nhadamard_transform(right)\nphase(0.19*left+0.23*right+0.31,1)\n"
            + action
            + "\nwith adjoint():\n    "
            + action,
        )
    )
    phases = torch.tensor(
        [
            cmath.exp(1j * (0.19 * left + 0.23 * right + 0.31))
            for left, right in product(range(4), range(8))
        ],
        dtype=torch.complex128,
    )
    reference = fq.Circuit(5, dtype=torch.complex128)
    for bit in range(5):
        reference.h(bit)
    reference.any(*range(5), matrix=torch.diag(phases))
    compare_states(program, reference, ("left", "right"))


@pytest.mark.parametrize("action", ("q += q", "update(q[0:2],q[1:3])", "q -= q[0:1]"))
def test_overlapping_physical_views_are_refused(action: str) -> None:
    with pytest.raises(HybridVerificationError, match="disjoint|overlap"):
        synth.synthesize(
            definition(
                "q:Output[QNum[3]]",
                "allocate(q)\n" + action,
                "@qfunc\ndef update(left:QNum,right:QNum):\n    right += left\n",
            )
        )


def test_const_accumulator_cannot_be_written() -> None:
    with pytest.raises(HybridVerificationError, match="writable"):
        synth.synthesize(
            definition(
                "left:Output[QNum[2]],right:Output[QNum[3]]",
                "allocate(left)\nallocate(right)\nupdate(left,right)",
                "@qfunc\ndef update(left:QNum,right:Const[QNum]):\n    right += left\n",
            )
        )


@pytest.mark.parametrize("limit", ("qubits", "operations", "memory"))
def test_existing_resource_ceilings_still_reject(limit: str) -> None:
    with pytest.raises(ValueError, match="budget|candidate|memory|limit"):
        synth.synthesize(
            definition(
                "left:Output[QNum[2]],right:Output[QNum[3]]",
                "allocate(left)\nallocate(right)\nright += left",
            ),
            max_qubits=4 if limit == "qubits" else 8,
            max_operations=1 if limit == "operations" else 2048,
            memory_limit_bytes=1 if limit == "memory" else 16 * 1024**2,
        )


@pytest.mark.parametrize("metadata", ("3,True,0", "3,False,1"))
@pytest.mark.parametrize("numeric_source", (False, True))
def test_signed_and_fractional_words_are_not_raw_modular_updates(
    metadata: str, numeric_source: bool
) -> None:
    left = metadata if numeric_source else "3"
    right = "3" if numeric_source else metadata
    with pytest.raises(HybridVerificationError, match="raw unsigned"):
        synth.synthesize(
            definition(
                f"left:Output[QNum[{left}]],right:Output[QNum[{right}]]",
                "allocate(left)\nallocate(right)\nright += left",
            )
        )


@pytest.mark.parametrize("preparation", ("allocate(left)", "allocate(right)"))
def test_both_registers_must_be_initialized(preparation: str) -> None:
    with pytest.raises(HybridVerificationError, match="uninitialized"):
        synth.synthesize(
            definition(
                "left:Output[QNum[2]],right:Output[QNum[3]]",
                preparation + "\nright += left",
            )
        )


@pytest.mark.parametrize(
    "source", ("left+1", "left*left", "[left[0],left[0]]", "left[0:0]", "left[0:4]")
)
def test_compound_duplicate_empty_and_out_of_bounds_addends_refuse(source: str) -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "left:Output[QNum[3]],right:Output[QNum[3]]",
                "allocate(left)\nallocate(right)\nright += " + source,
            )
        )


def test_permuted_source_view_keeps_explicit_low_bit_order() -> None:
    program = synth.synthesize(
        definition(
            "left:Output[QNum[3]],right:Output[QNum[3]]",
            "allocate(left)\nallocate(right)\nright += [left[2],left[0],left[1]]",
        )
    )
    wires = (program.registers["left"], program.registers["right"])
    for left, right in product(range(8), repeat=2):
        addend = ((left >> 2) & 1) | ((left & 1) << 1) | (((left >> 1) & 1) << 2)
        assert classical_apply(program.circuit, encode((left, right), wires)) == encode(
            (left, (right + addend) % 8), wires
        )


def test_two_register_controlled_helper_remains_refused() -> None:
    with pytest.raises(HybridVerificationError, match="one raw inout"):
        synth.synthesize(
            definition(
                "left:Output[QNum[2]],right:Output[QNum[3]],flag:Output[QBit]",
                "allocate(left)\nallocate(right)\nallocate(flag)\ncontrol(flag,lambda:update(left,right))",
                "@qfunc\ndef update(left:Const[QNum],right:QNum):\n    right += left\n",
            )
        )


def test_same_owner_const_and_writable_helper_views_keep_existing_refusal() -> None:
    with pytest.raises(HybridVerificationError, match="separate registers"):
        synth.synthesize(
            definition(
                "q:Output[QNum[5]]",
                "allocate(q)\nupdate(q[0:2],q[2:5])",
                "@qfunc\ndef update(left:Const[QNum],right:QNum):\n    right += left\n",
            )
        )


def test_direct_indexed_augmented_assignment_keeps_existing_refusal() -> None:
    with pytest.raises(HybridVerificationError, match="program.statement"):
        synth.synthesize(
            definition("q:Output[QNum[5]]", "allocate(q)\nq[2:5] += q[0:2]")
        )
