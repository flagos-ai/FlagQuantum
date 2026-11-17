"""Record constant field scaling and source rejection cases from frozen C++."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from kickmix_source_oracle import source_oracle

SOURCE_BLOB = "7b679a3e90bf211e432c95336599250ffd233d3b"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        (n, d, f)
        for n in (1, 2, 3, 4, 8, 17)
        for d in (0, 1)
        for f in sorted(
            {0, 1, 2, 3, 1 << n, (1 << n) | 1, (1 << (2 * n)) + 5, (1 << n) - 1}
        )
    ]
    cases.extend((256, d, 0xDEADBEEFCAFEF00D1234567890ABCDEF) for d in (0, 1))
    rows = []
    with source_oracle(
        args.source_root,
        "gf_arithmetic/gen_gf_imul_classical.cc",
        SOURCE_BLOB,
        Path(__file__).with_name("kickmix_field_constant_scale_file_oracle.cc"),
        dependencies=("gf_arithmetic/gen_linear_map.cc",),
    ) as oracle:
        for width, divide, factor in cases:
            raw = subprocess.check_output(
                [str(oracle), str(width), str(divide), hex(factor)],
                text=True,
                timeout=8,
            )
            modulus, *lines = raw.splitlines()
            rejected = lines == ["REJECTED"]
            records = [
                line
                for line in lines
                if not line.startswith(("APPEND_TO_REGISTER ", "REGISTER "))
            ]
            rows.append(
                {
                    "width": width,
                    "divide": bool(divide),
                    "factor": hex(factor),
                    "modulus": modulus,
                    "rejected": rejected,
                    "counts": {}
                    if rejected
                    else dict(Counter(line.split()[0] for line in records)),
                    "sha256": None
                    if rejected
                    else hashlib.sha256("\n".join(records).encode()).hexdigest(),
                }
            )
    args.output.write_text(
        json.dumps({"source_blob": SOURCE_BLOB, "cases": rows}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
