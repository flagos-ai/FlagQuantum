"""Verify locked physical field inversion and measured cleanup."""

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
from flagquantum.compiler.registers.finite_field.inverse_chain import (
    field_inverse_chain_size,
    field_inverse_workspace_size,
    physical_field_inverse,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import expand_reset_branches, forced_branch


class _Case(TypedDict):
    width: int
    route: int
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
    .joinpath("kickmix_field_inverse_file.json")
    .read_text()
)


@pytest.mark.parametrize(
    "case",
    _SOURCE["cases"],
    ids=lambda case: f"{case['width']}-{case['route']}",
)
def test_complete_frozen_inverse_streams(case: _Case) -> None:
    assert _SOURCE["source_blob"] == "3988c793db4c65454b555548107c77ab2e3c531f"
    n, route = case["width"], case["route"]
    work_size = field_inverse_chain_size if route >= 2 else field_inverse_workspace_size
    assert work_size(n) == case["work"]
    start = 2 * n

    def word(offset: int) -> tuple[int, ...]:
        return tuple(reversed(range(offset, offset + n)))

    workspace = tuple(
        q for offset in range(start, start + case["work"], n) for q in word(offset)
    )
    program, calls = physical_field_inverse(
        word(0),
        word(n),
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


@pytest.mark.parametrize(
    "width,retain,round_trip,events",
    [
        (2, False, True, 2),
        (3, False, False, 6),
        (3, True, True, 6),
    ],
)
def test_physical_inverse_all_reset_and_measurement_histories(
    width: int,
    retain: bool,
    round_trip: bool,
    events: int,
) -> None:
    work = (field_inverse_chain_size if retain else field_inverse_workspace_size)(width)
    n = 2 * width + work
    source = tuple(range(width))
    target = tuple(range(width, 2 * width))
    workspace = tuple(range(2 * width, n))
    programs = [
        physical_field_inverse(
            source,
            target,
            workspace,
            modulus=7 if width == 2 else 11,
            retain_chain=retain,
            erase=erase,
        )[0]
        for erase in ((False, True) if round_trip else (False,))
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
    indices = torch.arange(1 << width)
    amplitudes = torch.complex((indices + 1).double(), (7 - indices).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << n, dtype=torch.complex128)
    initial[indices << (work + width)] = amplitudes
    expected = initial
    if not round_trip:
        inverse = torch.tensor([0, 1, 5, 6, 7, 2, 3, 4])
        expected = torch.zeros_like(initial)
        expected[((indices << width) | inverse) << work] = amplitudes
    for outcomes in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, gates, outcomes),
            expected / 2 ** (events / 2),
            atol=5e-14,
            rtol=5e-14,
        )


def test_leased_workspace_preserves_unused_wires() -> None:
    args = ((0,), (1,), (2,))
    program, _ = physical_field_inverse(*args, modulus=3)
    assert program.n_qubits == 3
    assert all(2 not in op.qubits for op in program.operations)
    with pytest.raises(ValueError, match="workspace"):
        physical_field_inverse(*args, modulus=3, retain_chain=True)
