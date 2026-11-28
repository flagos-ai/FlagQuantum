"""Observe complete shared-span storage and draw position in frozen source."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from generate_striped_modular_sampling import SOURCE_BLOBS


def regenerate(source_root: Path, output: Path) -> int:
    for name, expected in SOURCE_BLOBS.items():
        raw = (source_root / "kickmix/util" / name).read_bytes()
        actual = hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen source blob mismatch for {name}")
    cases = []
    for width, modulus in ((2, 3), (4, 9), (8, 129), (17, 65537)):
        for delta in sorted({-width, 1 - width, -1, 0, 1, width - 1, width}):
            for seed in range(8):
                for repeats in (1, 3):
                    cases.append(
                        [
                            width,
                            modulus,
                            seed,
                            2 + max(0, -delta),
                            2 + max(0, delta),
                            repeats,
                        ]
                    )
    for delta in (-1, 0, 1):
        cases.append([1, 1, 0, 2 + max(0, -delta), 2 + max(0, delta), 1])
    with tempfile.TemporaryDirectory(prefix="fq-striped-alias-") as directory:
        binary = Path(directory) / "oracle"
        subprocess.run(
            [
                "clang++",
                "-std=c++20",
                "-O1",
                "-fsanitize=address,undefined",
                "-fno-sanitize-recover=all",
                "-I" + str(source_root),
                str(Path(__file__).with_name("striped_sampling_alias_oracle.cc")),
                str(source_root / "kickmix/util/mod_rand.cc"),
                str(source_root / "kickmix/util/fixed_width_int.cc"),
                "-o",
                str(binary),
            ],
            check=True,
            timeout=60,
        )
        result = subprocess.run(
            [str(binary)],
            input="".join(" ".join(map(str, case)) + "\n" for case in cases),
            text=True,
            capture_output=True,
            check=True,
            timeout=60,
        )
    if result.stderr:
        raise ValueError("Source sanitizer reported diagnostics: " + result.stderr)
    lines = result.stdout.splitlines()
    if len(lines) != len(cases):
        raise ValueError("Incomplete source observations")
    output.write_text(
        "".join(
            json.dumps([case, line], separators=(",", ":")) + "\n"
            for case, line in zip(cases, lines, strict=True)
        )
    )
    return len(cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(regenerate(args.source_root, args.output))
