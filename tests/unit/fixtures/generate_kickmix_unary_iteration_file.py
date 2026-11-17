"""Record marked traversal streams from the unchanged Kickmix cursor."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from itertools import product
from pathlib import Path

from kickmix_source_oracle import source_oracle

SOURCE_BLOB = "ec97900dbce6ac31ca66a33eb63e95ea7073fb1d"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases: list[tuple[int, int, tuple[int, ...], bool]] = []
    for width, mode in product((0, 1, 2, 3, 10, 63), range(4)):
        limit = 1 << width
        cases.extend(
            (width, mode, values, False)
            for values in ((), (limit, limit + 1), (limit - 1, 0, 0, limit, 0))
        )
    cases.append((10, 1, (), True))
    rows = []
    with source_oracle(
        args.source_root,
        "iteration/gen_unary_iteration.cc",
        SOURCE_BLOB,
        Path(__file__).with_name("kickmix_unary_iteration_file_oracle.cc"),
    ) as oracle:
        for width, mode, values, sweep in cases:
            raw = subprocess.check_output(
                [
                    str(oracle),
                    str(width),
                    str(mode),
                    "sweep" if sweep else "visits",
                    *map(str, values),
                ],
                text=True,
                timeout=8,
            )
            records = [
                line
                for line in raw.splitlines()
                if not line.startswith(("APPEND_TO_REGISTER ", "REGISTER "))
            ]
            rows.append(
                {
                    "width": width,
                    "mode": mode,
                    "values": values,
                    "sweep": sweep,
                    "counts": dict(Counter(line.split()[0] for line in records)),
                    "sha256": hashlib.sha256("\n".join(records).encode()).hexdigest(),
                }
            )
    args.output.write_text(
        json.dumps({"source_blob": SOURCE_BLOB, "cases": rows}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
