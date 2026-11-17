"""Record full physical doubling workloads from the unchanged source generator."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from kickmix_source_oracle import source_oracle

_SOURCE_BLOB = "f2c07c7825e05cd44bd2f793c0c8a826cf426cb5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = {}
    with source_oracle(
        args.source_root,
        "modular_arithmetic/gen_imul2_mod.cc",
        _SOURCE_BLOB,
        Path(__file__).with_name("kickmix_modular_doubling_workloads_oracle.cc"),
        dependencies=(
            "adders/gen_iadd.cc",
            "adders/gen_iadd1.cc",
            "adders/gen_iadd_classical.cc",
            "adders/gen_iadd_control.cc",
            "bits/gen_bit_rotate.cc",
            "comparators/gen_cmp.cc",
            "comparators/gen_cmp_low_space.cc",
            "comparators/gen_cmp_qq.cc",
        ),
    ) as oracle:
        for kind in ("const", "bits"):
            raw = subprocess.check_output([str(oracle), kind], text=True, timeout=8)
            records = [
                line
                for line in raw.splitlines()
                if not line.startswith(("APPEND_TO_REGISTER ", "REGISTER "))
            ]
            cases[kind] = {
                "counts": dict(Counter(line.split()[0] for line in records)),
                "records": len(records),
                "sha256": hashlib.sha256("\n".join(records).encode()).hexdigest(),
            }
    args.output.write_text(
        json.dumps({"source_blob": _SOURCE_BLOB, "cases": cases}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
