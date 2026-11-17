"""Record full division streams from the unchanged locked C++ source."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from itertools import product
from pathlib import Path

from kickmix_source_oracle import source_oracle

SOURCE_BLOB = "55de676044e1b676430c70304d3d03894974311f"
DEPENDENCIES = (
    "gen_gf_inverse",
    "gen_gf_mul",
    "gen_gf_iadd",
    "gen_gf_ifrobenius",
    "gen_gf_imul_classical",
    "gen_gf_imul_x",
    "gen_linear_map",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with source_oracle(
        args.source_root,
        "gf_arithmetic/gen_gf_div.cc",
        SOURCE_BLOB,
        Path(__file__).with_name("kickmix_field_division_file_oracle.cc"),
        dependencies=tuple(f"gf_arithmetic/{name}.cc" for name in DEPENDENCIES),
    ) as oracle:
        cases = (*product((1, 2, 4, 8), range(4), (0, 1)), (128, 0, 0), (128, 1, 0))
        for width, route, shared in cases:
            raw = subprocess.check_output(
                [str(oracle), str(width), str(route), str(shared)],
                text=True,
                timeout=15,
            )
            modulus, work, *records = raw.splitlines()
            lines = [
                line
                for line in records
                if not line.startswith(("APPEND_TO_REGISTER ", "REGISTER "))
            ]
            rows.append(
                {
                    "width": width,
                    "route": route,
                    "shared": bool(shared),
                    "modulus": modulus,
                    "work": int(work),
                    "records": len(lines),
                    "counts": dict(Counter(line.split()[0] for line in lines)),
                    "sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
                }
            )
    args.output.write_text(
        json.dumps({"source_blob": SOURCE_BLOB, "cases": rows}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
