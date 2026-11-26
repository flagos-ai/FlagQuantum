"""Regenerate unchanged main dispatch observations after frozen include checks."""

import argparse
import csv
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path


def regenerate(source: Path, inventory: Path, output: Path) -> None:
    source = source.resolve()
    rows = list(csv.DictReader(inventory.read_text(encoding="utf-8-sig").splitlines()))
    expected = {
        row["path"].removeprefix("kickmix/src/"): row["git_blob"]
        for row in rows
        if row["path"].startswith("kickmix/src/")
    }
    pending = [source / "kickmix/main.cc"]
    visited = set()
    system_headers = set()
    while pending:
        path = pending.pop()
        relative = path.relative_to(source).as_posix()
        if relative in visited:
            continue
        visited.add(relative)
        raw = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if blob != expected[relative]:
            raise ValueError(f"Frozen command dependency differs for {relative}")
        for include in re.findall(rb'#include\s+"([^"]+)"', raw):
            local = path.parent / include.decode()
            candidate = local if local.is_file() else source / include.decode()
            if candidate.is_file():
                pending.append(candidate.resolve())
            elif include.decode() in ("immintrin.h", "arm_neon.h"):
                system_headers.add(include.decode())
            else:
                raise FileNotFoundError(
                    f"Missing original dependency {include.decode()}"
                )
    with tempfile.TemporaryDirectory(prefix="fq-main-command-") as directory:
        binary = Path(directory) / "oracle"
        subprocess.run(
            [
                "clang++",
                "-std=c++20",
                "-O1",
                "-I" + str(source),
                str(Path(__file__).with_name("main_command_oracle.cc")),
                "-o",
                str(binary),
            ],
            check=True,
            capture_output=True,
            timeout=90,
        )
        names = (
            "circuit",
            "count_operations",
            "kmb2kmx",
            "kmx2kmb",
            "count_operations_sampled",
            "sample",
            "spin",
            "diagram",
        )
        cases = [[], [""], ["unknown"], ["--help"], ["CIRCUIT"], ["circuitx"]]
        cases += [
            [name, *rest]
            for name in names
            for rest in (
                [],
                ["--in", "x y", "--out=z"],
                ["throw"],
                ["", "é", "--", "-1", "line\nbreak"],
            )
        ]
        records = []
        for args in cases:
            result = subprocess.run(
                [str(binary), *args],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            records.append([args, result.returncode, result.stdout, result.stderr])
        output.write_text(
            json.dumps(
                {
                    "frozen_blobs": {name: expected[name] for name in sorted(visited)},
                    "system_headers": sorted(system_headers),
                    "commands": records,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    regenerate(args.source_root, args.inventory, args.output)


if __name__ == "__main__":
    main()
