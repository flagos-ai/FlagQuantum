"""Verify the full physical stream of each locked lookup workload."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from benchmarks.benchmark_physical_lookup import locked_lookup_workload
from flagquantum.compiler.formats.kmx_block_marks import (
    decode_kmx_block_marks,
    encode_kmx_block_marks,
)


@pytest.mark.parametrize(("kind", "bits"), [("lookup", 7681), ("erase", 7937)])
def test_locked_lookup_workload(kind: str, bits: int) -> None:
    fixture = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("kickmix_lookup_workloads.json")
        .read_text()
    )
    assert fixture["source_blob"] == "d527eeeddf06c10ffe69ab6408594c60fd248182"
    expected = fixture["cases"][kind]
    program, calls = locked_lookup_workload().generate(erase=kind == "erase")
    assert program.n_qubits == 48
    assert program.n_bits == bits
    assert program.input_bits == tuple(range(7680))
    text = encode_kmx_block_marks(program, calls)
    records = text.splitlines()[1:]
    assert len(records) == expected["records"]
    assert Counter(line.split()[0] for line in records) == expected["counts"]
    assert hashlib.sha256("\n".join(records).encode()).hexdigest() == expected["sha256"]
    assert decode_kmx_block_marks(text) == (program, calls)
