"""Capture complete register-data field, equality and diagnostic semantics."""

import argparse
import csv
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path


def regenerate(source: Path, inventory: Path, output: Path) -> None:
    src = (source / "kickmix/src").resolve()
    expected = {
        row["源文件"]: row["源对象"]
        for row in csv.DictReader(
            inventory.read_text(encoding="utf-8-sig").splitlines()
        )
    }
    pending = [
        src / "kickmix/circuit/register_data.h",
        src / "kickmix/id/qubit_id.cc",
        src / "kickmix/id/bit_id.cc",
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
            raise ValueError(f"Frozen register source differs for {relative}")
        verified[relative] = blob
        for include in re.findall(r'^#include "([^"]+)"', raw.decode(), re.M):
            pending.append(
                src / include
                if include.startswith("kickmix/")
                else path.parent / include
            )
    with tempfile.TemporaryDirectory(prefix="fq-register-data-") as folder:
        binary = Path(folder) / "probe"
        subprocess.run(
            [
                "clang++",
                "-std=c++20",
                "-O1",
                "-I" + str(src),
                str(Path(__file__).with_name("register_data.cc")),
                str(src / "kickmix/id/qubit_id.cc"),
                str(src / "kickmix/id/bit_id.cc"),
                "-o",
                str(binary),
            ],
            check=True,
            capture_output=True,
            timeout=90,
        )
        rows = subprocess.run(
            [str(binary)], check=True, text=True, capture_output=True, timeout=10
        ).stdout.splitlines()
    output.write_text(
        json.dumps(
            {
                "pin": "9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453",
                "frozen_blobs": verified,
                "source_rows": rows,
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
    args = parser.parse_args()
    regenerate(args.source, args.inventory, args.output)
