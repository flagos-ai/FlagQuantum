"""Regenerate complete frozen physical statistics without altering the source."""

import argparse
import csv
import doctest
import hashlib
import importlib
import json
import re
from pathlib import Path
from typing import Any, TypedDict


class StatisticsRow(TypedDict):
    items: list[list[str | int]]
    values: list[int]
    keys: list[str]
    instructions: int
    toffoli: int
    t_like: int
    non_clifford: int
    rotations: int
    levels: list[list[int]]
    representation: str


def snapshot(stats: Any) -> StatisticsRow:
    """Read the external binding once into primitive, owned observation fields."""
    return {
        "items": [[name, value] for name, value in stats.items()],
        "keys": list(stats.keys()),
        "values": list(stats.values()),
        "instructions": stats.num_instructions,
        "toffoli": stats.num_toffoli_like,
        "t_like": stats.num_t_like,
        "non_clifford": stats.num_non_clifford,
        "rotations": stats.num_z_pow(),
        "levels": [
            [level, count]
            for level, count in stats.z_pow_clifford_hierarchy_counts.items()
        ],
        "representation": repr(stats).replace(
            "km.OperationStatistics", "PhysicalOperationStatistics"
        ),
    }


def verify_source(source: Path, inventory: Path) -> dict[str, str]:
    expected = {
        row["path"].removeprefix("kickmix/src/"): row["git_blob"]
        for row in csv.DictReader(
            inventory.read_text(encoding="utf-8-sig").splitlines()
        )
        if row["path"].startswith("kickmix/src/")
    }
    pending = [
        source / "kickmix/py/operation_statistics.pybind.cc",
        source / "kickmix/py/operation_statistics.pybind.h",
    ]
    verified: dict[str, str] = {}
    while pending:
        path = pending.pop().resolve()
        name = path.relative_to(source).as_posix()
        if name in verified:
            continue
        raw = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if blob != expected[name]:
            raise ValueError(f"Frozen statistics source differs for {name}")
        verified[name] = blob
        for raw_include in re.findall(rb'#include\s+"([^"]+)"', raw):
            include = raw_include.decode()
            candidate = path.parent / include
            if not candidate.is_file():
                candidate = source / include
            if candidate.is_file():
                pending.append(candidate)
            elif include not in ("immintrin.h", "arm_neon.h"):
                raise FileNotFoundError(f"Missing original include {include}")
    return dict(sorted(verified.items()))


def regenerate(source: Path, inventory: Path, output: Path) -> None:
    verified = verify_source(source.resolve(), inventory)
    km = importlib.import_module("kickmix")
    if km.__version__ != "0.1" or not hasattr(km, "Simulator"):
        raise RuntimeError(
            "The locked native Kickmix extension and Simulator are required"
        )
    runner = doctest.DocTestRunner()
    parser = doctest.DocTestParser()
    for name, member in [
        ("OperationStatistics", km.OperationStatistics),
        *km.OperationStatistics.__dict__.items(),
    ]:
        if member.__doc__:
            runner.run(
                parser.get_doctest(
                    member.__doc__, {"km": km}, name, "original statistics binding", 0
                )
            )
    docs = runner.summarize()
    if docs.failed or not docs.attempted:
        raise AssertionError(docs)
    phases = ["0.0", "1.0"] + ["0." + str(5**k).zfill(k) for k in range(1, 128)]
    all_levels = "\n".join(
        "Z_POW q0 " + phase + (" if b0" if k % 2 else "")
        for k, phase in enumerate(phases)
    )
    all_operations = """NEG
BIT_INVERT b2
BIT_STORE0 b3
BIT_STORE1 b4
X q0
Z q0
R q0
HMR q1 b2
CX q0 q1
CZ q0 q1
SWAP q0 q1
CSWAP q0 q1 q2
CCX q0 q1 q2
CCZ q0 q1 q2
Z_POW q0 0.25
DEBUG_PRINT
DEBUG_PRINT q0
DEBUG_PRINT b0
"""
    mixed = (
        all_operations
        + "\n".join(line + " if b0" for line in all_operations.splitlines())
        + '\nPUSH_CONDITION if b0\nINFO_TAG "tag"\nPOP_CONDITION\n'
    )
    texts = [
        "",
        "X q0",
        "X q0 if b0",
        "Z_POW q0 0.25",
        "Z_POW q0 0.125",
        all_levels,
        mixed,
    ]
    statistics = km.Circuit(all_levels).operation_statistics()
    ranges = [
        [lo, hi, statistics.num_z_pow(lo, hi)]
        for lo in (-(1 << 63), -1, 0, 1, 2, 3, 127, 128, 129, (1 << 63) - 1)
        for hi in (None, -1, 0, 2, 3, 127, 128, 129, (1 << 63) - 1)
    ]
    batches = []
    for size in (0, 1, 3, 4, 64, 255, 256, 257, 300):
        words = [
            sum(1 << k for k in range(size) if k % 2 == 0),
            sum(1 << k for k in range(size) if k % 4 < 2),
        ]
        sim = km.Simulator(
            batch_size=size, counting_operations=True, debug_print=False, seed=1
        )
        for index, word in enumerate(words):
            sim.write_across_shots(km.b(index), word)
        # Measurement destinations stay independent of the classical feedback inputs.
        text = mixed.replace("HMR q1 b2", "HMR q1 b5")
        text += (
            "Z_POW q0 0.125 if b1\nZ_POW q0 0.0\nZ_POW q0 "
            + phases[-1]
            + ' if b0\nPUSH_CONDITION if b0\nPUSH_CONDITION if b1\nCCX q0 q1 q2\nZ_POW q0 0.125\nPOP_CONDITION\nINFO_TAG "nested"\nPOP_CONDITION\n'
        )
        sim.do(km.Circuit(text))
        unique: list[StatisticsRow] = []
        indices: list[int] = []
        for lane in range(size):
            observed = snapshot(sim.operation_statistics(lane))
            if observed not in unique:
                unique.append(observed)
            indices.append(unique.index(observed))
        batches.append(
            {
                "size": size,
                "words": words,
                "kmx": text,
                "total": snapshot(sim.operation_statistics()),
                "lane_snapshots": unique,
                "lane_indices": indices,
            }
        )
    sim = km.Simulator(
        batch_size=4, counting_operations=True, debug_print=False, seed=1
    )
    sim.write_across_shots(km.b(0), 5)
    sim.write_across_shots(km.b(1), 3)
    sim.do(km.Circuit(all_levels))
    batches.append(
        {
            "size": 4,
            "words": [5, 3],
            "kmx": all_levels,
            "total": snapshot(sim.operation_statistics()),
            "lane_snapshots": [snapshot(sim.operation_statistics(k)) for k in range(4)],
            "lane_indices": [0, 1, 2, 3],
        }
    )
    output.write_text(
        json.dumps(
            {
                "frozen_blobs": verified,
                "pin": "9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453",
                "source_doc_examples": docs.attempted,
                "static": [
                    [text, snapshot(km.Circuit(text).operation_statistics())]
                    for text in texts
                ],
                "ranges": ranges,
                "batches": batches,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    regenerate(args.source_root, args.inventory, args.output)


if __name__ == "__main__":
    main()
