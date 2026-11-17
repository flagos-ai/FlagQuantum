"""Retain the exact marked Fourier measurement streams through width 130."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from flagquantum.compiler.formats.kmx_block_marks import (
    decode_kmx_block_marks,
    encode_kmx_block_marks,
)
from flagquantum.compiler.registers.phase.qft_measurement import (
    physical_qft_measurement,
)


def test_locked_marked_fourier_streams() -> None:
    fixture = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("kickmix_qft_measurement_file.json")
        .read_text()
    )
    assert fixture["source_blob"] == "59c942ad6d1a4e40e3776166afb99220bbf93f6b"
    assert len(fixture["cases"]) == 66
    for case in fixture["cases"]:
        width: int = case["width"]
        mode = case["mode"]
        if mode == 0:
            source_targets = tuple(range(width))
            source_outputs = source_targets
        elif mode == 1:
            source_targets = tuple(reversed(range(width)))
            source_outputs = tuple(range(width))
        else:
            source_targets = tuple(range(1, 3 * width, 3))
            source_outputs = tuple(reversed(range(2, 2 * width + 2, 2)))
        program, calls = physical_qft_measurement(
            source_targets[::-1],
            source_outputs[::-1],
            n_qubits=max(1, width if mode < 2 else 3 * width + 2),
            inverse_qft=case["inverse"],
        )
        text = encode_kmx_block_marks(program, calls)
        records = text.splitlines()[1:]
        assert Counter(line.split()[0] for line in records) == case["counts"], case
        assert len(records) == case["records"], case
        assert (
            hashlib.sha256("\n".join(records).encode()).hexdigest() == case["sha256"]
        ), case
        assert decode_kmx_block_marks(text) == (program, calls)
