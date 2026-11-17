"""Capture the locked raw workload counts and ordered feedback digest."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from kickmix_modular_feedback import feedback_digest
from kickmix_source_oracle import source_oracle

_SOURCE_BLOB = "ba7e69f03ad225a13ca996bb44f23c36a6f4f28c"
_DEPENDENCIES = (
    "adders/gen_iadd.cc",
    "adders/gen_iadd1.cc",
    "adders/gen_iadd_classical.cc",
    "adders/gen_iadd_control.cc",
    "comparators/gen_cmp.cc",
    "comparators/gen_cmp_low_space.cc",
    "comparators/gen_cmp_qq.cc",
)


def source_fixture(oracle: Path) -> dict[str, object]:
    cases: dict[str, object] = {}
    for kind in ("add", "sub", "approx"):
        raw = subprocess.check_output([str(oracle), kind], text=True, timeout=8)
        records = [
            line
            for line in raw.splitlines()
            if not line.startswith(("APPEND_TO_REGISTER ", "REGISTER "))
        ]
        inputs = () if kind == "approx" else tuple(range(1, 255))
        effects, digest = feedback_digest(records, inputs)
        cases[kind] = {
            "counts": dict(Counter(line.split()[0] for line in records)),
            "records": len(records),
            "raw_sha256": hashlib.sha256("\n".join(records).encode()).hexdigest(),
            "feedback_effects": effects,
            "feedback_sha256": digest,
        }
    return {"source_blob": _SOURCE_BLOB, "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with source_oracle(
        args.source_root,
        "modular_arithmetic/gen_iadd_mod.cc",
        _SOURCE_BLOB,
        Path(__file__).with_name("kickmix_modular_addition_workloads_oracle.cc"),
        dependencies=_DEPENDENCIES,
    ) as oracle:
        fixture = source_fixture(oracle)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")


if __name__ == "__main__":
    main()
