"""Certify marked carry-controlled streams and physical measurement branches."""

import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import TypedDict

import pytest
import torch

from benchmarks import benchmark_physical_carry_controlled_addition as benchmark
from flagquantum.compiler.formats.kmx_block_marks import (
    decode_kmx_block_marks,
    encode_kmx_block_marks,
)
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.arithmetic.carry_controlled_addition import (
    physical_carry_controlled_addition,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from tests.unit.measurement_checks import expand_reset_branches, forced_branch


class _Source(TypedDict):
    width: int
    offset_tail: int
    extra_clean: int
    carry_kind: int
    subtract: bool
    sha256: str
    opcodes: dict[str, int]


def test_complete_locked_physical_streams() -> None:
    fixture = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("kickmix_controlled_addition_file.json")
        .read_text()
    )
    assert fixture["source_blob"] == "ee8062ae063ce241383bdb633ead74c4dafc5db1"
    cases: list[_Source] = fixture["cases"]
    assert len(cases) == 181
    for source in cases:
        n = source["width"]
        m = n + source["offset_tail"]
        perf = n == 256
        control = n + m + int(perf)
        carry = (False, True, n + m + int(not perf), ClassicalFieldCoefficient(0))[
            source["carry_kind"]
        ]
        clean_start = n + m + 2
        clean_stop = clean_start + (n if n >= 2 else 0) + source["extra_clean"]
        program, calls = physical_carry_controlled_addition(
            tuple(reversed(range(n))),
            tuple(reversed(range(n, n + m))),
            control=control,
            carry_in=carry,
            clean=tuple(range(clean_start, clean_stop)),
            measurement_bit=int(not perf),
            subtract=source["subtract"],
        )
        text = encode_kmx_block_marks(program, calls)
        lines = text.splitlines()[1:]
        assert (
            hashlib.sha256("\n".join(lines).encode()).hexdigest() == source["sha256"]
        ), source
        assert Counter(line.split()[0] for line in lines) == source["opcodes"]
        assert decode_kmx_block_marks(text) == (program, calls)
        if clean_stop > clean_start:
            assert program.n_qubits == clean_stop


@pytest.mark.parametrize(
    "kind,carry_value,subtract",
    [(0, 0, True), (2, 0, False), (3, 0, False), (3, 1, False)],
)
def test_physical_reset_and_feedback_branches(
    kind: int, carry_value: int, subtract: bool
) -> None:
    target, offset, clean = (0, 1), (2, 3), (6, 7)
    carry = (False, True, 5, ClassicalFieldCoefficient(3))[kind]
    program, _ = physical_carry_controlled_addition(
        target,
        offset,
        control=4,
        carry_in=carry,
        clean=clean,
        measurement_bit=1,
        subtract=subtract,
    )
    # Carry b3 is an external input above the HMR destination b1.
    setup = (ClassicalXor((), 3, bool(carry_value)).instruction(),) if kind == 3 else ()
    expected_inputs = (3,) if kind == 3 else ()
    assert program.input_bits == expected_inputs
    gates = expand_reset_branches(
        (*setup, *bind_reversible_phase_execution(program)),
        first_slot=program.n_bits + 1,
    )
    indices = torch.arange(64)
    amplitudes = torch.complex((indices + 1).double(), (5 - indices).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(256, dtype=torch.complex128)
    initial[indices << 2] = amplitudes
    rhs = (indices >> 2) & 3
    delta = -rhs if subtract else rhs + (indices & 1 if kind == 2 else carry_value)
    lhs = ((indices >> 4) + ((indices >> 1) & 1) * delta) % 4
    expected = torch.zeros_like(initial)
    expected[((lhs << 4) | (indices & 15)) << 2] = amplitudes
    assert sum(gate.name == "measure" for gate in gates) == 2
    for outcomes in product((0, 1), repeat=2):
        torch.testing.assert_close(
            forced_branch(initial, gates, outcomes),
            expected / 2,
            atol=5e-14,
            rtol=5e-14,
        )


def test_native_benchmark_uses_locked_physical_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark, "perf_counter_ns", iter((0, 1000)).__next__)
    row = benchmark.benchmark_carry_controlled_addition(warmup=0, repeats=1)
    assert row["width"] == 256
    assert row["records_including_marks"] == 2810
    assert row["median_seconds"] == 1e-6
    assert row["primitive_counts"] == {
        "cx": 1277,
        "ccx": 511,
        "r": 255,
        "hmr": 255,
        "cz": 255,
        "z": 255,
    }
