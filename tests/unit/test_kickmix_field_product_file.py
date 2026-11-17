"""Check standalone binary product traces and all small measurement histories."""

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
from flagquantum.compiler.registers.finite_field.polynomial_product import (
    physical_field_product,
    physical_polynomial_product,
)
from flagquantum.compiler.registers.finite_field.product_erasure import (
    physical_field_product_phase,
)
from flagquantum.compiler.registers.primitives.measurement import measure_x_clear_ir
from flagquantum.core.classical_dataflow import ClassicalXor
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
    .joinpath("kickmix_field_product_file.json")
    .read_text()
)


@pytest.mark.parametrize(
    "case", _SOURCE["cases"], ids=lambda row: f"{row['width']}-{row['route']}"
)
def test_complete_frozen_product_streams(case: _Case) -> None:
    assert _SOURCE["source_blob"] == "d4448e4c8216fc5ba82c3c492b94ea18e31e3799"
    width, route = case["width"], case["route"]
    modulus = int(case["modulus"], 0)
    off = 2 * width - 1 if route == 0 else 0 if route >= 5 else width
    target = tuple(reversed(range(off)))
    left = tuple(reversed(range(off, off + width)))
    right = tuple(reversed(range(off + width, off + 2 * width)))
    if route == 0:
        program, calls = physical_polynomial_product(left, right, target)
    elif route >= 5:
        mask = (0,) * width if route == 6 else tuple(reversed(range(width)))
        program, calls = physical_field_product_phase(
            left, right, mask, modulus=modulus
        )
    else:
        control = off + 2 * width if route in (3, 4) else None
        work = (
            tuple(reversed(range(control + 1, control + 1 + case["work"])))
            if control is not None
            else ()
        )
        program, calls = physical_field_product(
            left,
            right,
            target,
            modulus=modulus,
            erase=route in (2, 4),
            control=control,
            workspace=work,
        )
    text = encode_kmx_block_marks(program, calls)
    body = text.partition("\n")[2].rstrip("\n")
    records = body.splitlines()
    assert len(records) == case["records"]
    assert Counter(line.split()[0] for line in records) == case["counts"]
    assert hashlib.sha256(body.encode()).hexdigest() == case["sha256"]
    if width <= 17:
        assert decode_kmx_block_marks(text) == (program, calls)


def _gf4_products(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    a, b = left >> 1, left & 1
    c, d = right >> 1, right & 1
    high = (a & c) ^ (a & d) ^ (b & c)
    low = (a & c) ^ (b & d)
    return (high << 1) | low


def test_controlled_unmultiply_arbitrary_target_and_all_histories() -> None:
    program, _ = physical_field_product(
        (0, 1),
        (2, 3),
        (4, 5),
        control=6,
        workspace=(7, 8),
        modulus=7,
        erase=True,
    )
    gates = (
        *(ClassicalXor((), bit).instruction() for bit in program.input_bits),
        *bind_reversible_phase_execution(program),
    )
    gates = expand_reset_branches(gates, first_slot=program.n_bits + 1)
    assert sum(gate.name == "measure" for gate in gates) == 4
    words = torch.arange(128)
    amplitudes = torch.complex((words + 1).double(), (7 - words).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(512, dtype=torch.complex128)
    initial[words << 2] = amplitudes
    product_words = _gf4_products(words >> 5, (words >> 3) & 3)
    expected = torch.zeros_like(initial)
    mapped = words ^ (((words & 1) * product_words) << 1)
    expected[mapped << 2] = amplitudes
    for outcomes in product((0, 1), repeat=4):
        torch.testing.assert_close(
            forced_branch(initial, gates, outcomes),
            expected / 4,
            atol=5e-14,
            rtol=5e-14,
        )


def test_aliased_phase_mask_keeps_source_toggles_and_every_branch() -> None:
    program, _ = physical_field_product_phase((0, 1), (2, 3), (0, 0), modulus=7)
    gates = (
        *measure_x_clear_ir(4, 0),
        *(
            ClassicalXor((), bit).instruction()
            for bit in program.input_bits
            if bit != 0
        ),
        *bind_reversible_phase_execution(program),
    )
    words = torch.arange(16)
    amplitudes = torch.complex((words + 1).double(), (7 - words).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(32, dtype=torch.complex128)
    initial[words << 1] = amplitudes
    product_words = _gf4_products(words >> 2, words & 3)
    parity = (product_words >> 1) ^ (product_words & 1)
    for outcome in (0, 1):
        expected = torch.zeros_like(initial)
        expected[words << 1] = amplitudes * (1 - 2 * outcome * parity) / 2**0.5
        torch.testing.assert_close(
            forced_branch(initial, gates, (outcome,)),
            expected,
            atol=5e-14,
            rtol=5e-14,
        )


def test_empty_polynomial_product_has_no_call_records() -> None:
    program, calls = physical_polynomial_product((), (), ())
    assert not program.operations and not calls
