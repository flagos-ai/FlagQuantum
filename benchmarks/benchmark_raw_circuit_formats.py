# SPDX-License-Identifier: Apache-2.0
# Workloads adapted from the locked Kickmix mutable_circuit.perf.cc and circuit.perf.cc.

"""Benchmark native raw circuit operations on the paired KMX and KMB fixtures."""

import argparse
import json
import platform
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import BinaryIO

from flagquantum.compiler.formats.kmb import encode_kmb
from flagquantum.compiler.formats.kmx_builder import KmxCircuitBuilder
from flagquantum.compiler.formats.kmx_circuit import read_kmb_circuit, read_kmx_circuit

_SCOPE_BLOCK = "PUSH_CONDITION if b0\nPOP_CONDITION\nX q0\nPUSH_CONDITION if b1\n"
_SCOPE_COUNT = 1001


def _measure(
    action: Callable[[], int], expected: int, warmup: int, repeats: int
) -> float:
    def verify(value: int) -> None:
        if value != expected:
            raise RuntimeError(f"Expected benchmark result {expected}, got {value}")

    for _ in range(warmup):
        verify(action())
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        result = action()
        samples.append(perf_counter() - start)
        verify(result)
    return median(samples)


def benchmark_circuits(
    fixture_dir: Path, *, warmup: int = 1, repeats: int = 3
) -> dict[str, object]:
    """Verify the paired records, then time the six source workload boundaries."""
    if warmup < 0 or repeats < 1:
        raise ValueError("Warmup must be nonnegative and repeats must be positive")
    kmx_path = fixture_dir / "big_circuit.kmx"
    kmb_path = fixture_dir / "big_circuit.kmb"
    kmx_raw, kmb_raw = kmx_path.read_bytes(), kmb_path.read_bytes()
    text = kmx_raw.decode()

    canonical = read_kmb_circuit(kmb_raw)
    builder = KmxCircuitBuilder()
    builder.append_kmx_text(text)
    from_text = builder.freeze()
    with kmx_path.open("rb") as source:
        builder.clear()
        builder.append_kmx_file(source)
    if (
        not canonical.records
        or from_text != canonical
        or builder.freeze() != canonical
        or encode_kmb(canonical.to_kmx()) != kmb_raw
    ):
        raise ValueError("KMX and KMB fixtures must encode the same canonical circuit")
    records = len(canonical.records)
    scope_circuit = read_kmx_circuit(_SCOPE_BLOCK * _SCOPE_COUNT)
    rows: dict[str, object] = {}

    def record(
        name: str,
        operation: Callable[[], int],
        measured_records: int,
        expected: int,
        description: str,
        byte_count: int | None = None,
    ) -> None:
        seconds = _measure(operation, expected, warmup, repeats)
        rows[name] = {
            "median_seconds": seconds,
            "records": measured_records,
            "records_per_second": measured_records / seconds,
            "bytes": byte_count,
            "bytes_per_second": (
                byte_count / seconds if byte_count is not None else None
            ),
            "timed_boundary": description,
        }

    def append_text() -> int:
        builder.clear()
        builder.append_kmx_text(text)
        return builder.operation_count

    def append_file(source: BinaryIO, *, freeze: bool) -> int:
        source.seek(0)
        builder.clear()
        builder.append_kmx_file(source)
        return len(builder.freeze().records) if freeze else builder.operation_count

    def read_binary(source: BinaryIO) -> int:
        source.seek(0)
        return len(read_kmb_circuit(source.read()).records)

    record(
        "mutable_circuit_append_from_kmx_text",
        append_text,
        records,
        records,
        "clear and append predecoded KMX text",
        len(kmx_raw),
    )
    record(
        "circuit_max_push_condition_depth",
        scope_circuit.max_condition_depth,
        len(scope_circuit.records),
        _SCOPE_COUNT,
        "scan already frozen records",
    )
    with kmx_path.open("rb") as source, kmb_path.open("rb") as binary:
        record(
            "mutable_circuit_append_from_kmx_file",
            lambda: append_file(source, freeze=False),
            records,
            records,
            "rewind, clear and append binary KMX stream",
            len(kmx_raw),
        )
        record(
            "mutable_circuit_from_kmb_file",
            lambda: read_binary(binary),
            records,
            records,
            "rewind, read, validate and freeze KMB",
            len(kmb_raw),
        )
        record(
            "circuit_from_kmx",
            lambda: append_file(source, freeze=True),
            records,
            records,
            "rewind, clear, append and freeze KMX",
            len(kmx_raw),
        )
        record(
            "circuit_from_kmb",
            lambda: read_binary(binary),
            records,
            records,
            "rewind, read, validate and freeze KMB",
            len(kmb_raw),
        )

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "warmup": warmup,
        "repeats": repeats,
        "kmx_sha256": sha256(kmx_raw).hexdigest(),
        "kmb_sha256": sha256(kmb_raw).hexdigest(),
        "scope_depth": _SCOPE_COUNT,
        "workloads": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_dir", type=Path)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    result = benchmark_circuits(
        args.fixture_dir, warmup=args.warmup, repeats=args.repeats
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
