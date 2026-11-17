"""Record standalone forward and adjoint Frobenius calls from locked C++."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from kickmix_source_oracle import source_oracle

SOURCE_BLOB = "b4edd6b7339591e8831db6d1cea4781314f09a94"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        (n, k, a)
        for n in (1, 2, 3, 4, 8, 17)
        for k in sorted({0, 1, 2, n - 1, n, n + 1, 129, (1 << 64) - 1})
        for a in (0, 1)
    ]
    cases.extend(((256, 1, 0), (256, 128, 0)))
    rows = []
    with source_oracle(
        args.source_root,
        "gf_arithmetic/gen_gf_ifrobenius.cc",
        SOURCE_BLOB,
        Path(__file__).with_name("kickmix_field_frobenius_file_oracle.cc"),
        dependencies=("gf_arithmetic/gen_linear_map.cc",),
    ) as oracle:
        for n, k, a in cases:
            raw = subprocess.check_output(
                [str(oracle), str(n), str(k), str(a)],
                text=True,
                timeout=8,
            )
            modulus, *lines = raw.splitlines()
            records = [
                line
                for line in lines
                if not line.startswith(("APPEND_TO_REGISTER ", "REGISTER "))
            ]
            rows.append(
                {
                    "width": n,
                    "power": k,
                    "adjoint": a,
                    "modulus": modulus,
                    "counts": dict(Counter(line.split()[0] for line in records)),
                    "sha256": hashlib.sha256("\n".join(records).encode()).hexdigest(),
                }
            )
    args.output.write_text(
        json.dumps({"source_blob": SOURCE_BLOB, "cases": rows}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
