"""Check the standalone Kickmix Frobenius and exact adjoint register calls."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import TypedDict

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.formats.kmx_block_marks import (
    decode_kmx_block_marks,
    encode_kmx_block_marks,
)
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.finite_field.binary_linear_map import (
    physical_field_frobenius,
)


class _Case(TypedDict):
    width: int
    power: int
    adjoint: int
    modulus: str
    counts: dict[str, int]
    sha256: str


def test_complete_frozen_forward_and_adjoint_streams() -> None:
    fixture = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("kickmix_field_frobenius_file.json")
        .read_text()
    )
    assert fixture["source_blob"] == "b4edd6b7339591e8831db6d1cea4781314f09a94"
    cases: list[_Case] = fixture["cases"]
    assert len(cases) == 86
    for case in cases:
        width, power = case["width"], case["power"]
        program, calls = physical_field_frobenius(
            tuple(reversed(range(width))),
            power,
            modulus=int(case["modulus"], 0),
            adjoint=bool(case["adjoint"]),
        )
        assert program.n_qubits == width
        assert program.n_bits == 0
        assert all(op.opcode in ("cx", "swap") for op in program.operations)
        if power % width == 0:
            assert not calls and not program.operations
        text = encode_kmx_block_marks(program, calls)
        lines = text.splitlines()[1:]
        assert Counter(line.split()[0] for line in lines) == case["counts"], case
        assert (
            hashlib.sha256("\n".join(lines).encode()).hexdigest() == case["sha256"]
        ), case
        assert decode_kmx_block_marks(text) == (program, calls)


@pytest.mark.parametrize("power", [0, 1, 3, 4, 129, (1 << 64) - 1])
def test_frobenius_physical_complex_state_with_spectator(power: int) -> None:
    width, modulus = 4, 0x13
    target = (4, 2, 1, 0)
    total = 5
    basis = torch.arange(1 << total, dtype=torch.float64)
    state = torch.complex(basis + 1, 3 * basis - 2)
    state /= state.norm()
    expected = torch.empty_like(state)
    masks = tuple(1 << (total - wire - 1) for wire in target)
    for initial in range(1 << total):
        value = sum(
            bool(initial & bit) << (width - i - 1) for i, bit in enumerate(masks)
        )
        for _ in range(power % width):
            value = sum(((value >> i) & 1) << (2 * i) for i in range(width))
            while value.bit_length() > width:
                value ^= modulus << (value.bit_length() - width - 1)
        mapped = initial & ~sum(masks)
        mapped |= sum(
            mask for i, mask in enumerate(masks) if value & (1 << (width - i - 1))
        )
        expected[mapped] = state[initial]
    fwd, _ = physical_field_frobenius(target, power, modulus=modulus)
    adj, _ = physical_field_frobenius(target, power, modulus=modulus, adjoint=True)
    circuit = Circuit(total, inputs=state, dtype=torch.complex128)
    transformed = circuit.append(
        FieldOperation(total, bind_reversible_phase_execution(fwd))
    )
    torch.testing.assert_close(
        transformed.state().flatten(), expected, rtol=0, atol=1e-12
    )
    restored = transformed.append(
        FieldOperation(total, bind_reversible_phase_execution(adj))
    )
    torch.testing.assert_close(restored.state().flatten(), state, rtol=0, atol=1e-12)


def test_frobenius_physical_rejects_bad_bindings() -> None:
    with pytest.raises(ValueError, match="register degree"):
        physical_field_frobenius((0, 1), modulus=0x13)
    with pytest.raises(ValueError, match="nonnegative count"):
        physical_field_frobenius((0, 1, 2, 3), -1, modulus=0x13)
