"""Compare the complete locked physical increment streams and execution."""

import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import TypedDict

import pytest
import torch

from benchmarks.benchmark_physical_increment import benchmark_increment
from flagquantum.compiler.formats.kmx_block_marks import encode_kmx_block_marks
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.arithmetic.increment import physical_increment
from tests.unit.measurement_checks import expand_reset_branches, forced_branch


class _Source(TypedDict):
    width: int
    controlled: bool
    clean: int
    sha256: str
    opcodes: dict[str, int]


_SOURCE: list[_Source] = json.loads(
    (Path(__file__).parent / "fixtures" / "kickmix_increment_file.json").read_text()
)


def test_full_physical_increment_fingerprints() -> None:
    for source in _SOURCE:
        n = source["width"]
        control: int | bool = n if source["controlled"] else True
        result, calls = physical_increment(
            tuple(reversed(range(n))),
            control=control,
            clean=tuple(range(n + 1, n + 1 + source["clean"])),
        )
        lines = encode_kmx_block_marks(result, calls).splitlines()[1:]
        assert hashlib.sha256("\n".join(lines).encode()).hexdigest() == source["sha256"]
        assert Counter(line.split()[0] for line in lines) == source["opcodes"]


@pytest.mark.parametrize("controlled", (False, True))
def test_physical_reset_and_carry_branches(controlled: bool) -> None:
    n = 4
    logical = n + int(controlled)
    clean_count = n - (2 if controlled else 3)
    width = logical + clean_count
    result, _ = physical_increment(
        tuple(range(n)),
        control=n if controlled else True,
        clean=tuple(range(logical, width)),
    )
    states = torch.arange(1 << logical)
    amplitudes = torch.complex((3 + states).double(), (4 - states).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    initial[states << clean_count] = amplitudes
    expected = torch.zeros_like(initial)
    if controlled:
        target = ((states >> 1) + (states & 1)) % (1 << n)
        output = (target << 1) | (states & 1)
    else:
        output = (states + 1) % (1 << n)
    expected[output << clean_count] = amplitudes
    gates = expand_reset_branches(
        bind_reversible_phase_execution(result), first_slot=result.n_bits + 1
    )
    events = 2 * clean_count
    assert sum(gate.name == "measure" for gate in gates) == events
    for outcomes in product((0, 1), repeat=events):
        actual = forced_branch(initial, gates, outcomes)
        torch.testing.assert_close(
            actual, expected / 2 ** (events / 2), atol=5e-14, rtol=5e-14
        )


def test_both_native_benchmark_workloads(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmarks import benchmark_physical_increment as module

    clock = iter((0, 1000, 1000, 2000))
    monkeypatch.setattr(module, "perf_counter_ns", clock.__next__)
    report = benchmark_increment(warmup=0, repeats=1)
    for name, records in (("unconditional", 1270), ("controlled", 1274)):
        assert isinstance(row := report[name], dict)
        assert row["width"] == 256
        assert row["records_including_marks"] == records
        assert row["median_seconds"] == 1e-6
        assert row["records_per_second"] == records * 1e6
