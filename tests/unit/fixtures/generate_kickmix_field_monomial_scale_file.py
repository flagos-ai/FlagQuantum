"""Record signed monomial transforms from the frozen C++ generation paths."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import TypedDict

from kickmix_source_oracle import source_oracle

SOURCE_BLOB = "efc026a507d520c924034adfa2c42569211cfa8d"


class _Record(TypedDict):
    width: int
    divide: int
    power: int
    modulus: str
    counts: dict[str, int]
    sha256: str


def _read(oracle: Path, width: int, divide: int, power: int) -> _Record:
    raw = subprocess.check_output(
        [str(oracle), str(width), str(divide), str(power)],
        text=True,
        timeout=8,
    )
    modulus, *lines = raw.splitlines()
    records = [
        line
        for line in lines
        if not line.startswith(("APPEND_TO_REGISTER ", "REGISTER "))
    ]
    return {
        "width": width,
        "divide": divide,
        "power": power,
        "modulus": modulus,
        "counts": dict(Counter(line.split()[0] for line in records)),
        "sha256": hashlib.sha256("\n".join(records).encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[_Record] = []
    with source_oracle(
        args.source_root,
        "gf_arithmetic/gen_gf_imul_x.cc",
        SOURCE_BLOB,
        Path(__file__).with_name("kickmix_field_monomial_scale_file_oracle.cc"),
        dependencies=("gf_arithmetic/gen_linear_map.cc",),
    ) as oracle:
        for width in (1, 2, 3, 4, 8, 17):
            modulus = int(_read(oracle, width, 0, 0)["modulus"], 0)
            threshold = width * width // (modulus.bit_count() - 1)
            powers = {0, 1, width, width + 1, threshold, threshold + 1, (1 << 64) - 1}
            for power in sorted(powers):
                rows.extend(_read(oracle, width, divide, power) for divide in (0, 1))
        rows.extend(_read(oracle, 512, 0, power) for power in (1, 512))
    args.output.write_text(
        json.dumps({"source_blob": SOURCE_BLOB, "cases": rows}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
