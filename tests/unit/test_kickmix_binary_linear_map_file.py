"""Verify complete source PLU traces, inverse cases, and complex states."""

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
    _frobenius_columns,
    physical_binary_linear_map,
)


class _Case(TypedDict):
    width: int
    mode: int
    adjoint: bool
    columns: list[str]
    rejected: bool
    counts: dict[str, int]
    sha256: str


def test_complete_source_linear_map_and_adjoints() -> None:
    fixture = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("kickmix_binary_linear_map_file.json")
        .read_text()
    )
    assert fixture["source_blob"] == "33e44b9d7508129c1360a750522cca383f46864a"
    cases: list[_Case] = fixture["cases"]
    assert len(cases) == 58
    assert sum(case["rejected"] for case in cases) == 12
    for case in cases:
        width = case["width"]
        target = tuple(reversed(range(width)))
        columns = tuple(int(value, 0) for value in case["columns"])
        if case["rejected"]:
            with pytest.raises(ValueError, match="invertible"):
                physical_binary_linear_map(target, columns, adjoint=case["adjoint"])
            continue
        if case["mode"] == 2:
            modulus = (1 << width) | (0x1B if width == 64 else 0x425)
            assert _frobenius_columns(width, 1, modulus) == columns
        program, calls = physical_binary_linear_map(
            target, columns, adjoint=case["adjoint"]
        )
        assert program.n_qubits == max(1, width)
        assert program.n_bits == 0
        assert all(op.opcode in ("cx", "swap") for op in program.operations)
        text = encode_kmx_block_marks(program, calls)
        lines = text.splitlines()[1:]
        assert Counter(line.split()[0] for line in lines) == case["counts"], case
        assert (
            hashlib.sha256("\n".join(lines).encode()).hexdigest() == case["sha256"]
        ), case
        assert decode_kmx_block_marks(text) == (program, calls)


@pytest.mark.parametrize("columns", [(1, 2, 4, 8), (8, 2, 4, 1), (9, 4, 5, 6)])
def test_strided_linear_map_preserves_all_complex_amplitudes(
    columns: tuple[int, ...],
) -> None:
    target = (4, 2, 1, 0)
    total = 5
    basis = torch.arange(1 << total, dtype=torch.float64)
    state = torch.complex(basis + 1, 3 * basis - 2)
    state /= state.norm()
    expected = torch.empty_like(state)
    masks = tuple(1 << (total - wire - 1) for wire in target[::-1])
    for initial in range(1 << total):
        word = 0
        for column, mask in zip(columns, masks, strict=True):
            if initial & mask:
                word ^= column
        mapped = initial & ~sum(masks)
        mapped |= sum(mask for i, mask in enumerate(masks) if word & (1 << i))
        expected[mapped] = state[initial]
    forward, _ = physical_binary_linear_map(target, columns)
    adjoint, _ = physical_binary_linear_map(target, columns, adjoint=True)
    circuit = Circuit(total, inputs=state, dtype=torch.complex128)
    transformed = circuit.append(
        FieldOperation(total, bind_reversible_phase_execution(forward))
    )
    torch.testing.assert_close(
        transformed.state().flatten(), expected, rtol=0, atol=1e-12
    )
    restored = transformed.append(
        FieldOperation(total, bind_reversible_phase_execution(adjoint))
    )
    torch.testing.assert_close(restored.state().flatten(), state, rtol=0, atol=1e-12)


@pytest.mark.parametrize("columns", [(1,), (1, 2, 4), (1, 4)])
def test_rejects_matrix_shape_and_entries(columns: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="square register map"):
        physical_binary_linear_map((0, 1), columns)
