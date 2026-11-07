"""Regenerate controlled carry-addition fixtures from unchanged locked source."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from kickmix_source_oracle import normalized_stream, source_oracle

SOURCE_BLOB = "ee8062ae063ce241383bdb633ead74c4dafc5db1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    harness = Path(__file__).with_name("kickmix_controlled_addition_oracle.cc")
    with source_oracle(
        args.source_root, "adders/gen_iadd_control.cc", SOURCE_BLOB, harness
    ) as oracle:
        for width in range(9):
            for tail in (0, 2):
                for extra in (0, 2):
                    for kind, subtract in (
                        (0, False),
                        (1, False),
                        (2, False),
                        (3, False),
                        (0, True),
                    ):
                        text = subprocess.check_output(
                            [
                                str(oracle),
                                str(width),
                                str(tail),
                                str(extra),
                                str(kind),
                                str(int(subtract)),
                            ],
                            text=True,
                            timeout=5,
                        )
                        stream = normalized_stream(text)
                        cases.append(
                            {
                                "width": width,
                                "offset_tail": tail,
                                "extra_clean": extra,
                                "carry_kind": kind,
                                "subtract": subtract,
                                "sha256": hashlib.sha256(
                                    json.dumps(stream).encode()
                                ).hexdigest(),
                                "opcodes": dict(
                                    Counter(
                                        primitive.split()[0] for primitive, _ in stream
                                    )
                                ),
                            }
                        )
    args.output.write_text(
        json.dumps(
            {
                "source_commit": "9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453",
                "source_blob": SOURCE_BLOB,
                "normalization": "Register/INFO annotations and reset of caller-supplied zero carries omitted; ordered primitive operands and effective classical guard sets retained.",
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
