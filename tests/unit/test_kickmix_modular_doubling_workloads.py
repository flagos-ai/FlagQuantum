"""Verify the full physical stream of each locked doubling workload."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from benchmarks.benchmark_physical_modular_doubling import locked_doubling_workloads
from flagquantum.compiler.formats.kmx_block_marks import (
    decode_kmx_block_marks,
    encode_kmx_block_marks,
)


@pytest.mark.parametrize(
    ("kind", "input_width", "scratch_limit"),
    [("const", 0, 61), ("bits", 254, 508)],
)
def test_locked_doubling_workload(
    kind: str, input_width: int, scratch_limit: int
) -> None:
    fixture = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("kickmix_modular_doubling_workloads.json")
        .read_text()
    )
    assert fixture["source_blob"] == "f2c07c7825e05cd44bd2f793c0c8a826cf426cb5"
    expected = fixture["cases"][kind]
    program, calls = locked_doubling_workloads()[kind]()
    assert program.n_qubits == 517
    assert program.n_bits == scratch_limit + 1
    # The source allocator expects these scratch slots to start in zero.
    # They are declared in the tagged transport alongside the actual inputs.
    assert program.input_bits == (
        *range(input_width),
        *range(input_width + 2, scratch_limit),
    )
    text = encode_kmx_block_marks(program, calls)
    records = text.splitlines()[1:]
    assert len(records) == expected["records"]
    assert Counter(line.split()[0] for line in records) == expected["counts"]
    assert hashlib.sha256("\n".join(records).encode()).hexdigest() == expected["sha256"]
    assert decode_kmx_block_marks(text) == (program, calls)
