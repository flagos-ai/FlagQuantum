"""Capture raw counts and symbolic feedback from the locked negation source."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from itertools import product
from pathlib import Path

from kickmix_modular_feedback import feedback_digest
from kickmix_source_oracle import source_oracle

_SOURCE_BLOB = "73017205939c5ca0803f216fa91c9252047af90d"
_DEPENDENCIES = (
    "adders/gen_iadd.cc",
    "adders/gen_iadd1.cc",
    "adders/gen_iadd_classical.cc",
    "adders/gen_iadd_control.cc",
    "comparators/gen_cmp.cc",
    "comparators/gen_cmp_low_space.cc",
    "comparators/gen_cmp_qq.cc",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases: list[dict[str, object]] = []
    with source_oracle(
        args.source_root,
        "modular_arithmetic/gen_ineg_mod.cc",
        _SOURCE_BLOB,
        Path(__file__).with_name("kickmix_modular_negation_file_oracle.cc"),
        dependencies=_DEPENDENCIES,
    ) as oracle:
        for width, mode, dirty in product((1, 2, 4, 8, 17), range(3), (0, 8)):
            raw = subprocess.check_output(
                [str(oracle), str(width), str(mode), str(dirty)], text=True, timeout=8
            )
            records = [
                line
                for line in raw.splitlines()
                if not line.startswith(("APPEND_TO_REGISTER ", "REGISTER "))
            ]
            rejected = records == ["REJECTED"]
            inputs = tuple(range(width - 1)) if mode == 1 else ()
            effects, digest = (0, "") if rejected else feedback_digest(records, inputs)
            cases.append(
                {
                    "width": width,
                    "mode": mode,
                    "dirty": dirty,
                    "rejected": rejected,
                    "counts": dict(Counter(line.split()[0] for line in records)),
                    "records": len(records),
                    "raw_sha256": hashlib.sha256(
                        "\n".join(records).encode()
                    ).hexdigest(),
                    "feedback_effects": effects,
                    "feedback_sha256": digest,
                }
            )
    args.output.write_text(
        json.dumps({"source_blob": _SOURCE_BLOB, "cases": cases}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
