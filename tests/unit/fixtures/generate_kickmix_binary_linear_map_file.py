"""Record locked matrix inputs and the forward or inverse physical PLU trace."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from itertools import product
from pathlib import Path

from kickmix_source_oracle import source_oracle

SOURCE_BLOB = "33e44b9d7508129c1360a750522cca383f46864a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = (
        *product((0, 1, 2, 3, 4, 9, 17), (0, 1, 3, 4), (0, 1)),
        (64, 2, 0),
        (256, 2, 0),
    )
    rows = []
    with source_oracle(
        args.source_root,
        "gf_arithmetic/gen_linear_map.cc",
        SOURCE_BLOB,
        Path(__file__).with_name("kickmix_binary_linear_map_file_oracle.cc"),
    ) as oracle:
        for width, mode, adjoint in cases:
            raw = subprocess.check_output(
                [str(oracle), str(width), str(mode), str(adjoint)],
                text=True,
                timeout=8,
            )
            lines = raw.splitlines()
            columns = [hex(int(column[::-1], 2)) for column in lines[:width]]
            records = [
                line
                for line in lines[width:]
                if not line.startswith(("APPEND_TO_REGISTER ", "REGISTER "))
            ]
            rows.append(
                {
                    "width": width,
                    "mode": mode,
                    "adjoint": bool(adjoint),
                    "columns": columns,
                    "rejected": records == ["REJECTED"],
                    "counts": dict(Counter(line.split()[0] for line in records)),
                    "sha256": hashlib.sha256("\n".join(records).encode()).hexdigest(),
                }
            )
    args.output.write_text(
        json.dumps({"source_blob": SOURCE_BLOB, "cases": rows}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
