"""Verify unchanged comparator streams and native coherent physical execution."""

import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Literal

import pytest
import torch

from benchmarks import benchmark_physical_comparison as benchmark
from flagquantum.compiler.formats.kmx_block_marks import (
    decode_kmx_block_marks,
    encode_kmx_block_marks,
)
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.comparison.physical_comparison import (
    physical_comparison,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from tests.unit.measurement_checks import expand_reset_branches, forced_branch

_Relation = Literal["lt", "le", "gt", "ge", "eq"]
_Binding = tuple[int, int, str, _Relation, int, str, str, int, str]


def test_full_locked_comparison_streams() -> None:
    fixture = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("kickmix_comparison_file.json")
        .read_text()
    )
    assert fixture["source_blob"] == "fd73b653bd9ef09000e56021aab804a0b2733cf3"
    cases: list[_Binding] = fixture["cases"]
    assert len(cases) == 1262
    for n, work, kind, relation, phase, control_kind, tolerance, size, digest in cases:
        right: tuple[int | bool | ClassicalFieldCoefficient, ...]
        q_count = b_count = 0
        if kind == "q":
            q_count = n
            right = tuple(reversed(range(n, 2 * n)))
        elif kind == "b":
            b_count = n
            right = tuple(ClassicalFieldCoefficient(i) for i in reversed(range(n)))
        elif kind == "m":
            q_count, b_count = (n + 2) // 3, (n + 1) // 3
            right = tuple(
                n + i // 3
                if i % 3 == 0
                else ClassicalFieldCoefficient(i // 3)
                if i % 3 == 1
                else i % 2 == 0
                for i in reversed(range(n))
            )
        else:
            right = (True,) * (n - n // 2) + (False,) * (n // 2)
        cursor = n + q_count
        target = None if phase else cursor
        cursor += int(not phase)
        control = cursor if control_kind == "q" else True
        cursor += int(control_kind == "q")
        equal = ClassicalFieldCoefficient(b_count) if control_kind == "b" else False
        program, calls = physical_comparison(
            tuple(reversed(range(n))),
            right,
            target,
            tuple(range(cursor, cursor + work)),
            relation=relation,
            or_equal=equal,
            control=control,
            btol=float(tolerance),
        )
        text = encode_kmx_block_marks(program, calls)
        records = text.splitlines()[1:]
        assert len(records) == size, (
            n,
            work,
            kind,
            relation,
            phase,
            control_kind,
            tolerance,
        )
        assert hashlib.sha256("\n".join(records).encode()).hexdigest() == digest, (
            n,
            work,
            kind,
            relation,
            phase,
            control_kind,
            tolerance,
        )
        assert decode_kmx_block_marks(text) == (program, calls)


@pytest.mark.parametrize(
    "relation,phase,minimize",
    [
        ("lt", False, True),
        ("le", True, False),
        ("gt", True, False),
        ("ge", False, False),
        ("eq", True, False),
    ],
)
def test_physical_comparison_all_reset_and_hmr_branches(
    relation: _Relation, phase: bool, minimize: bool
) -> None:
    bits = tuple(ClassicalFieldCoefficient(i) for i in range(3))
    program, _ = physical_comparison(
        (0, 1, 2),
        bits,
        None if phase else 3,
        (4, 5, 6),
        relation=relation,
        minimize_qubits=minimize,
        or_equal=ClassicalFieldCoefficient(3) if relation == "gt" else False,
    )
    # RHS is 101. The greater-than case enables equality through an input bit.
    setup = tuple(
        ClassicalXor((), i, bool(value)).instruction()
        for i, value in enumerate((1, 0, 1, 1))
        if i in program.input_bits
    )
    gates = expand_reset_branches(
        (*setup, *bind_reversible_phase_execution(program)),
        first_slot=program.n_bits + len(program.operations) + 1,
    )
    indices = torch.arange(16)
    amplitudes = torch.complex((indices + 1).double(), (19 - indices).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(128, dtype=torch.complex128)
    initial[indices << 3] = amplitudes
    lhs = indices >> 1
    selected = {
        "lt": lhs < 5,
        "le": lhs <= 5,
        "gt": lhs >= 5,
        "ge": lhs >= 5,
        "eq": lhs == 5,
    }[relation]
    expected = torch.zeros_like(initial)
    if phase:
        expected[indices << 3] = amplitudes * (1 - 2 * selected.int())
    else:
        expected[(indices ^ selected.int()) << 3] = amplitudes
    events = sum(gate.name == "measure" for gate in gates)
    assert events > 0
    for outcomes in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, gates, outcomes),
            expected / 2 ** (events / 2),
            atol=5e-14,
            rtol=5e-14,
        )


def test_native_benchmarks_use_locked_layouts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        benchmark, "perf_counter_ns", iter((0, 1000, 2000, 3000)).__next__
    )
    result = benchmark.benchmark_comparison(warmup=0, repeats=1)
    for clean, records, ccx, hmr in (("256", 3834, 255, 255), ("3", 3140, 516, 17)):
        assert isinstance(row := result[clean], dict)
        assert row["records_including_marks"] == records
        assert row["median_seconds"] == 1e-6
        assert row["primitive_counts"]["ccx"] == ccx
        assert row["primitive_counts"]["hmr"] == hmr
