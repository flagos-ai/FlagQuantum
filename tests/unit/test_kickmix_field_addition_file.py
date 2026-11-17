"""Check whole physical addition records and coherent classical guard feedback."""

import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import TypedDict

import pytest
import torch

from benchmarks import benchmark_physical_field_addition as benchmark
from flagquantum.compiler.formats.kmx_block_marks import (
    decode_kmx_block_marks,
    encode_kmx_block_marks,
)
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient as Bit,
)
from flagquantum.compiler.registers.finite_field.physical_field_addition import (
    physical_field_addition,
)
from flagquantum.compiler.registers.primitives.measurement import measure_x_clear_ir
from tests.unit.measurement_checks import forced_branch


class _SourceCase(TypedDict):
    width: int
    kind: str
    control: str
    layout: int
    counts: dict[str, int]
    sha256: str


def test_complete_frozen_field_addition_streams() -> None:
    fixture = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("kickmix_field_addition_file.json")
        .read_text()
    )
    assert fixture["source_blob"] == "4fec5c24910e0cb9f605d2572dc8e334f1e0e59b"
    cases: list[_SourceCase] = fixture["cases"]
    assert len(cases) == 111
    for case in cases:
        n, kind, layout = case["width"], case["kind"], case["layout"]
        stride = 2 if layout == 2 else 1
        target = tuple(reversed(range(0, n * stride, stride)))
        quantum = n * stride
        classical = 0
        source: tuple[int | bool | Bit, ...]
        if kind == "q":
            source = tuple(range(quantum, 2 * quantum, stride))
            if layout != 1:
                source = source[::-1]
            quantum *= 2
        elif kind == "b":
            source = tuple(Bit(i) for i in reversed(range(n)))
            classical = n
        elif kind == "m":
            source = tuple(
                quantum + k // 4
                if k % 4 == 0
                else Bit(k // 4 % 2)
                if k % 4 == 1
                else k % 4 != 3
                for k in reversed(range(n))
            )
            quantum += (n + 3) // 4
            classical = 2
        else:
            source = tuple(kind != "z" and k % 4 != 3 for k in reversed(range(n)))
        controls: dict[str, int | bool | Bit | None] = {
            "q": quantum,
            "alias": Bit(0),
            "b": Bit(classical),
            "true": None,
            "false": False,
        }
        program, calls = physical_field_addition(
            source, target, control=controls[case["control"]]
        )
        text = encode_kmx_block_marks(program, calls)
        lines = text.splitlines()[1:]
        assert Counter(line.split()[0] for line in lines) == case["counts"], case
        assert (
            hashlib.sha256("\n".join(lines).encode()).hexdigest() == case["sha256"]
        ), case
        assert decode_kmx_block_marks(text) == (program, calls)
        assert not any(op.opcode in {"r", "hmr"} for op in program.operations)


@pytest.mark.parametrize("guard", [0, 1])
def test_physical_field_addition_preserves_all_guard_and_coefficient_histories(
    guard: int,
) -> None:
    program, _ = physical_field_addition(
        (0, Bit(1), True), (2, 3, 4), control=Bit(guard)
    )
    gates = (
        *measure_x_clear_ir(5, 0),
        *measure_x_clear_ir(5, 1),
        *bind_reversible_phase_execution(program),
    )
    basis = torch.arange(32)
    amplitudes = torch.complex((basis + 1).double(), (3 * basis - 5).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(64, dtype=torch.complex128)
    initial[::2] = amplitudes
    for outcomes in product((0, 1), repeat=2):
        enabled = outcomes[guard]
        offset = ((basis >> 4) << 2) | (outcomes[1] << 1) | 1
        expected = torch.zeros_like(initial)
        expected[(basis ^ (enabled * offset)) << 1] = amplitudes / 2
        torch.testing.assert_close(
            forced_branch(initial, gates, outcomes),
            expected,
            rtol=0,
            atol=1e-12,
        )


def test_native_benchmark_layouts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        benchmark, "perf_counter_ns", iter((0, 1000, 2000, 3000)).__next__
    )
    result = benchmark.benchmark_field_addition(warmup=0, repeats=1)
    for kind, opcode in (("uncontrolled", "cx"), ("controlled", "ccx")):
        assert isinstance(row := result[kind], dict)
        assert row["records_including_marks"] == 514
        assert row["primitive_counts"] == {opcode: 512}
        assert row["median_seconds"] == 1e-6
