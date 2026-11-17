"""Record marked Fourier measurement streams from the locked source."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from itertools import product
from pathlib import Path

from kickmix_source_oracle import source_oracle

_SOURCE_BLOB = "59c942ad6d1a4e40e3776166afb99220bbf93f6b"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    with source_oracle(
        args.source_root,
        "qft/gen_qft_demolition_measure.cc",
        _SOURCE_BLOB,
        Path(__file__).with_name("kickmix_qft_measurement_file_oracle.cc"),
    ) as oracle:
        widths = (0, 1, 2, 3, 4, 8, 17, 64, 128, 129, 130)
        for width, mode, inverse in product(widths, range(3), range(2)):
            raw = subprocess.check_output(
                [str(oracle), str(width), str(mode), str(inverse)],
                text=True,
                timeout=8,
            )
            records = raw.splitlines()
            cases.append(
                {
                    "width": width,
                    "mode": mode,
                    "inverse": bool(inverse),
                    "records": len(records),
                    "counts": dict(Counter(line.split()[0] for line in records)),
                    "sha256": hashlib.sha256("\n".join(records).encode()).hexdigest(),
                }
            )
    args.output.write_text(
        json.dumps({"source_blob": _SOURCE_BLOB, "cases": cases}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
