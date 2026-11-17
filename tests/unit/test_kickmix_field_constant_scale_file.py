"""Check constant multiply/divide against locked physical records."""

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
    physical_field_constant_scale,
)


class _Case(TypedDict):
    width: int
    divide: bool
    factor: str
    modulus: str
    rejected: bool
    counts: dict[str, int]
    sha256: str | None


def test_complete_frozen_constant_multiply_and_divide() -> None:
    fixture = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("kickmix_field_constant_scale_file.json")
        .read_text()
    )
    assert fixture["source_blob"] == "7b679a3e90bf211e432c95336599250ffd233d3b"
    cases: list[_Case] = fixture["cases"]
    assert len(cases) == 90
    assert sum(case["rejected"] for case in cases) == 22
    for case in cases:
        width = case["width"]
        target = tuple(reversed(range(width)))
        factor, modulus = int(case["factor"], 0), int(case["modulus"], 0)
        if case["rejected"]:
            with pytest.raises(ValueError, match="by zero"):
                physical_field_constant_scale(
                    target, factor, modulus=modulus, divide=case["divide"]
                )
            continue
        program, calls = physical_field_constant_scale(
            target, factor, modulus=modulus, divide=case["divide"]
        )
        assert program.n_qubits == width
        assert program.n_bits == 0
        assert all(op.opcode in ("cx", "swap") for op in program.operations)
        text = encode_kmx_block_marks(program, calls)
        lines = text.splitlines()[1:]
        assert Counter(line.split()[0] for line in lines) == case["counts"], case
        assert (
            hashlib.sha256("\n".join(lines).encode()).hexdigest() == case["sha256"]
        ), case
        assert decode_kmx_block_marks(text) == (program, calls)


@pytest.mark.parametrize("factor", [1, 0x12, 0x107, 0x124])
@pytest.mark.parametrize("divide", [False, True])
def test_constant_scale_complete_complex_state_and_spectator(
    factor: int, divide: bool
) -> None:
    width, modulus = 4, 0x13
    target = (4, 2, 1, 0)
    total = 5
    basis = torch.arange(1 << total, dtype=torch.float64)
    state = torch.complex(basis + 1, 3 * basis - 2)
    state /= state.norm()
    expected = torch.empty_like(state)
    masks = tuple(1 << (total - wire - 1) for wire in target)
    images = []
    for value in range(1 << width):
        result = 0
        for bit in range(width):
            if value & (1 << bit):
                result ^= factor << bit
        while result.bit_length() > width:
            result ^= modulus << (result.bit_length() - width - 1)
        images.append(result)
    for initial in range(1 << total):
        value = sum(
            bool(initial & bit) << (width - i - 1) for i, bit in enumerate(masks)
        )
        mapped_word = images.index(value) if divide else images[value]
        mapped = initial & ~sum(masks)
        mapped |= sum(
            mask for i, mask in enumerate(masks) if mapped_word & (1 << (width - i - 1))
        )
        expected[mapped] = state[initial]
    program, _ = physical_field_constant_scale(
        target, factor, modulus=modulus, divide=divide
    )
    circuit = Circuit(total, inputs=state, dtype=torch.complex128)
    transformed = circuit.append(
        FieldOperation(total, bind_reversible_phase_execution(program))
    )
    torch.testing.assert_close(
        transformed.state().flatten(), expected, rtol=0, atol=1e-12
    )
