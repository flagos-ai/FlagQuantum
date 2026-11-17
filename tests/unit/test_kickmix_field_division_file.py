"""Verify locked physical field-division records and quotient cleanup."""

import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import TypedDict

import pytest
import torch

from flagquantum.compiler.formats.kmx_block_marks import (
    decode_kmx_block_marks,
    encode_kmx_block_marks,
)
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.finite_field.division_chain import (
    field_division_workspace_size,
    physical_field_division,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import expand_reset_branches, forced_branch


class _Case(TypedDict):
    width: int
    route: int
    shared: bool
    modulus: str
    work: int
    records: int
    counts: dict[str, int]
    sha256: str


class _Fixture(TypedDict):
    source_blob: str
    cases: list[_Case]


_SOURCE: _Fixture = json.loads(
    Path(__file__)
    .with_name("fixtures")
    .joinpath("kickmix_field_division_file.json")
    .read_text()
)


@pytest.mark.parametrize(
    "case",
    _SOURCE["cases"],
    ids=lambda case: f"{case['width']}-{case['route']}-shared{int(case['shared'])}",
)
def test_complete_frozen_division_streams(case: _Case) -> None:
    assert _SOURCE["source_blob"] == "55de676044e1b676430c70304d3d03894974311f"
    n, route = case["width"], case["route"]
    assert field_division_workspace_size(n) == case["work"]
    target_start = n if case["shared"] else 2 * n
    start = target_start + n

    def word(offset: int) -> tuple[int, ...]:
        return tuple(reversed(range(offset, offset + n)))

    workspace = tuple(
        q for offset in range(start, start + case["work"], n) for q in word(offset)
    )
    program, calls = physical_field_division(
        word(0),
        word(0 if case["shared"] else n),
        word(target_start),
        workspace,
        modulus=int(case["modulus"], 0),
        retain_chain=route >= 2,
        erase=bool(route % 2),
    )
    text = encode_kmx_block_marks(program, calls)
    body = text.partition("\n")[2].rstrip("\n")
    records = body.splitlines()
    assert len(records) == case["records"]
    assert Counter(line.split()[0] for line in records) == case["counts"]
    assert hashlib.sha256(body.encode()).hexdigest() == case["sha256"]
    if n <= 8:
        assert decode_kmx_block_marks(text) == (program, calls)


@pytest.mark.parametrize("width,retain,events", [(1, False, 5), (2, True, 6)])
def test_every_physical_divide_undivide_branch(
    width: int, retain: bool, events: int
) -> None:
    # Coherent states cover the zero divisor and every possible operand.
    work = field_division_workspace_size(width)
    logical = 2 * width
    n = logical + width + work
    lhs = tuple(range(width))
    rhs = tuple(range(width, logical))
    target = tuple(range(logical, logical + width))
    workspace = tuple(range(logical + width, n))
    programs = [
        physical_field_division(
            lhs,
            rhs,
            target,
            workspace,
            modulus=3 if width == 1 else 7,
            retain_chain=retain,
            erase=erase,
        )[0]
        for erase in (False, True)
    ]
    gates: tuple[Instruction, ...] = tuple(
        gate
        for program in programs
        for gate in (
            *(ClassicalXor((), bit).instruction() for bit in program.input_bits),
            *bind_reversible_phase_execution(program),
        )
    )
    gates = expand_reset_branches(gates, first_slot=max(p.n_bits for p in programs) + 1)
    assert sum(gate.name == "measure" for gate in gates) == events
    indices = torch.arange(1 << logical)
    amplitudes = torch.complex((indices + 1).double(), (7 - indices).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << n, dtype=torch.complex128)
    initial[indices << (work + width)] = amplitudes
    for outcomes in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, gates, outcomes),
            initial / 2 ** (events / 2),
            atol=5e-14,
            rtol=5e-14,
        )


def test_leased_workspace_preserves_unused_wires() -> None:
    args = ((0,), (1,), (2,), (3, 4))
    program, _ = physical_field_division(*args, modulus=3)
    assert program.n_qubits == 5
    assert all(4 not in op.qubits for op in program.operations)
    with pytest.raises(ValueError, match="workspace"):
        physical_field_division(*args, modulus=3, retain_chain=True)
