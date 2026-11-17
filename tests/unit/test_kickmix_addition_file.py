"""Check the source in-place quantum arithmetic file and original diagrams."""

import json
from itertools import product
from pathlib import Path
from typing import TypedDict

import pytest
import torch

from flagquantum.compiler.formats.kmb import KmxRegister
from flagquantum.compiler.formats.kmx_block_marks import encode_kmx_block_marks
from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.arithmetic.physical_inplace_addition import (
    physical_inplace_addition,
)
from flagquantum.drawer.reversible_phase_diagram import render_circuit_text
from tests.unit.measurement_checks import expand_reset_branches, forced_branch


class _SourceCase(TypedDict):
    clean: int
    mode: str
    kmx: str
    diagram: str


_SOURCE: list[_SourceCase] = json.loads(
    (
        Path(__file__).parent / "fixtures" / "kickmix_inplace_addition_file.json"
    ).read_text()
)


@pytest.mark.parametrize(
    "case", _SOURCE, ids=lambda case: f"{case['mode']}-{case['clean']}"
)
def test_physical_register_add_sub_matches_original_layout_and_diagram(
    case: _SourceCase,
) -> None:
    target = tuple(range(1, 7))
    offset = tuple(range(7, 13))
    program, calls = physical_inplace_addition(
        target[::-1],
        offset[::-1],
        control=0,
        clean=tuple(range(13, 13 + case["clean"])),
        subtract=case["mode"] == "sub",
    )
    declarations = [
        line
        for i, wires in enumerate(((0,), target, offset))
        for line in KmxRegister("", tuple(("q", q) for q in wires)).to_kmx(i)
    ]
    _, _, body = encode_kmx_block_marks(program, calls).partition("\n")
    circuit = read_kmx_circuit("\n".join((*declarations, body)))
    assert circuit.to_kmx() == case["kmx"].rstrip("\n")
    assert render_circuit_text(circuit) == case["diagram"]


@pytest.mark.parametrize("subtract", [False, True])
@pytest.mark.parametrize(
    "n,m,clean_count,dirty_count,events",
    [
        (2, 1, 1, 0, 2),
        (3, 3, 0, 0, 0),
        (4, 3, 2, 0, 4),
        (4, 2, 1, 1, 0),
    ],
)
def test_every_physical_reset_and_carry_history(
    n: int, m: int, clean_count: int, dirty_count: int, events: int, subtract: bool
) -> None:
    # Put clean storage at the end and initialize all logical and dirty wires
    # in a complex superposition. Non-borrowed scratch remains zero.
    logical = n + m + 1 + dirty_count
    width = logical + clean_count
    target = tuple(reversed(range(n)))
    offset = tuple(reversed(range(n, n + m)))
    program, _ = physical_inplace_addition(
        target,
        offset,
        clean=tuple(range(logical, width)),
        dirty=tuple(range(n + m + 1, logical)),
        control=n + m,
        subtract=subtract,
    )
    inputs = torch.arange(1 << logical)
    psi = torch.complex((inputs + 1).double(), (5 - 2 * inputs).double())
    psi /= psi.norm()
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    basis = inputs << clean_count
    initial[basis] = psi

    def little_endian(wires: tuple[int, ...]) -> torch.Tensor:
        value = torch.zeros_like(basis)
        for lane, wire in enumerate(reversed(wires)):
            value |= ((basis >> (width - 1 - wire)) & 1) << lane
        return value

    left = little_endian(target)
    right = little_endian(offset)
    enable = (basis >> (width - 1 - n - m)) & 1
    updated = (left + (-right if subtract else right) * enable) % (1 << n)
    final = basis.clone()
    for lane, wire in enumerate(reversed(target)):
        final ^= (((left ^ updated) >> lane) & 1) << (width - 1 - wire)
    expected = torch.zeros_like(initial)
    expected[final] = psi

    gates = expand_reset_branches(
        bind_reversible_phase_execution(program), first_slot=program.n_bits + 1
    )
    assert sum(gate.name == "measure" for gate in gates) == events
    for outcomes in product((0, 1), repeat=events):
        actual = forced_branch(initial, gates, outcomes)
        torch.testing.assert_close(
            actual, expected / 2 ** (events / 2), atol=5e-14, rtol=5e-14
        )
