"""Regenerate increment stream digests from the locked, unmodified C++ source."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from kickmix_source_oracle import source_oracle

SOURCE_BLOB = "890b484fb6043d45d8e60a91dd241ab8092f7b21"
SOURCE_COMMIT = "9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, required=True, help="locked kickmix directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    harness = Path(__file__).with_name("kickmix_increment_oracle.cc")
    with source_oracle(
        args.source_root, "adders/gen_iadd1.cc", SOURCE_BLOB, harness
    ) as oracle:
        for width in range(17):
            for controlled in (False, True):
                for extra in (0, 2):
                    clean = max(0, width - (2 if controlled else 3)) + extra
                    text = subprocess.check_output(
                        [str(oracle), str(width), str(clean), str(int(controlled))],
                        text=True,
                        timeout=5,
                    )
                    lines = [
                        line
                        for line in text.splitlines()
                        if not line.startswith(
                            ("APPEND_TO_REGISTER ", "REGISTER ", "INFO_TAG ", "R ")
                        )
                    ]
                    cases.append(
                        {
                            "width": width,
                            "controlled": controlled,
                            "extra_clean": extra,
                            "sha256": hashlib.sha256(
                                "\n".join(lines).encode()
                            ).hexdigest(),
                            "opcodes": dict(Counter(line.split()[0] for line in lines)),
                        }
                    )
    args.output.write_text(
        json.dumps(
            {
                "source_commit": SOURCE_COMMIT,
                "source_blob": SOURCE_BLOB,
                "normalization": "Register/INFO metadata omitted; R on supplied clean wires is an explicit zero-input precondition. All other operations, operands and feedback slots retained.",
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
