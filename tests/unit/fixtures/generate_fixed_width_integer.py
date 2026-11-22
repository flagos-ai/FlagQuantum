"""Regenerate every scalar reference result using unchanged locked C++ sources."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from fixed_width_reference import read_cases

SOURCE_BLOBS = {
    "fixed_width_int.cc": "50a2300a311befb075bd6680287fa3e0e36bacc2",
    "fixed_width_int.h": "7057645bae714c72deb5b60dca85589bd4a9c8a7",
    "fixed_width_int.perf.cc": "7b915ff386a89038685827f3be1bcaadee0e91b4",
    "fixed_width_int.test.cc": "d4a35f0eb5049c2b68dc96d23baf9b68643ea790",
    "word_ops.h": "b0cab29ebcd7e616b6a40e7e4d99eb72bf0c9a45",
    "bit.h": "ea1c426299ab2ed0a7da662f629b133a106228e2",
}


def regenerate(source_root: Path, corpus: Path, output: Path) -> int:
    for name, expected in SOURCE_BLOBS.items():
        raw = (source_root / "kickmix/util" / name).read_bytes()
        actual = hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen source blob mismatch for {name}")
    cases = [case for case, _ in read_cases(corpus)]
    with tempfile.TemporaryDirectory(prefix="fq-fixed-width-reference-") as directory:
        binary = Path(directory) / "oracle"
        subprocess.run(
            [
                "clang++",
                "-std=c++20",
                "-O1",
                "-I" + str(source_root),
                str(Path(__file__).with_name("fixed_width_integer_oracle.cc")),
                str(source_root / "kickmix/util/fixed_width_int.cc"),
                "-o",
                str(binary),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        result = subprocess.run(
            [str(binary)],
            input="".join(case.source_input() for case in cases),
            text=True,
            capture_output=True,
            check=True,
            timeout=120,
        )
    lines = result.stdout.splitlines()
    if len(lines) != len(cases):
        raise ValueError("Source returned an incomplete reference stream")
    with output.open("w") as stream:
        for case, line in zip(cases, lines, strict=True):
            stream.write(json.dumps([case, line], separators=(",", ":")) + "\n")
    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        f"Recorded {regenerate(args.source_root, args.corpus, args.output)} original observations"
    )


if __name__ == "__main__":
    main()
