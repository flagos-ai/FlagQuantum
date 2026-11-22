"""Capture complete unchanged physical packed scalar lane semantics."""

import argparse
import csv
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path


def regenerate(
    source: Path, inventory: Path, output: Path, *, x86: bool = False
) -> None:
    src = (source / "kickmix/src").resolve()
    expected = {
        row["源文件"]: row["源对象"]
        for row in csv.DictReader(
            inventory.read_text(encoding="utf-8-sig").splitlines()
        )
    }
    pending = list((src / "kickmix/simd").glob("*.h")) + [
        src / "kickmix/simd/simd.test.cc",
        src / "test.test.cc",
    ]
    verified = {}
    while pending:
        path = pending.pop().resolve()
        relative = "kickmix/src/" + path.relative_to(src).as_posix()
        if relative in verified:
            continue
        raw = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if blob != expected[relative]:
            raise ValueError(f"Frozen SIMD source differs for {relative}")
        verified[relative] = blob
        for include in re.findall(r'^#include "([^"]+)"', raw.decode(), re.M):
            if include.startswith("gtest/") or include == "immintrin.h":
                continue
            local = path.parent / include
            pending.append(local if local.exists() else src / include)
    observations = {}
    for architecture in (("native", "x86_64") if x86 else ("native",)):
        with tempfile.TemporaryDirectory(prefix="fq-packed-word-") as folder:
            binary = Path(folder) / "probe"
            flags = ["-arch", "x86_64"] if architecture == "x86_64" else []
            subprocess.run(
                [
                    "clang++",
                    "-std=c++20",
                    "-O1",
                    *flags,
                    "-I" + str(src),
                    str(Path(__file__).with_name("packed_word.cc")),
                    "-o",
                    str(binary),
                ],
                check=True,
                capture_output=True,
                timeout=90,
            )
            observations[architecture] = subprocess.run(
                [str(binary)], check=True, text=True, capture_output=True, timeout=30
            ).stdout.splitlines()
    output.write_text(
        json.dumps(
            {
                "pin": "9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453",
                "frozen_blobs": verified,
                "source_rows": observations,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--x86", action="store_true")
    args = parser.parse_args()
    regenerate(args.source, args.inventory, args.output, x86=args.x86)
