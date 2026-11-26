"""Compile unchanged locked scalar identity files and record their observations."""

import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import TypedDict


class ScalarIdentityCorpus(TypedDict):
    pin: str
    frozen_blobs: dict[str, str]
    source_rows: list[str]


def regenerate(source: Path, inventory: Path, output: Path) -> None:
    expected = {
        row["path"]: row["git_blob"]
        for row in csv.DictReader(
            inventory.read_text(encoding="utf-8-sig").splitlines()
        )
    }
    names = ["id_tagging.h"] + [
        name + extension
        for name in ("bit_id", "qubit_id", "register_id", "xbit_id", "xbool")
        for extension in (".cc", ".h")
    ]
    verified = {}
    for name in names:
        relative = "kickmix/src/kickmix/id/" + name
        raw = (source / relative).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if blob != expected[relative]:
            raise ValueError(f"Frozen scalar identity source differs for {relative}")
        verified[relative] = blob
    with tempfile.TemporaryDirectory(prefix="fq-scalar-identity-") as folder:
        binary = Path(folder) / "oracle"
        subprocess.run(
            [
                "clang++",
                "-std=c++20",
                "-O1",
                "-I" + str(source / "kickmix/src"),
                str(Path(__file__).with_name("scalar_identities.cc")),
                *[
                    str(source / "kickmix/src/kickmix/id" / name)
                    for name in names
                    if name.endswith(".cc")
                ],
                "-o",
                str(binary),
            ],
            check=True,
            capture_output=True,
            timeout=90,
        )
        result = subprocess.run(
            [str(binary)], check=True, text=True, capture_output=True, timeout=10
        )
    corpus: ScalarIdentityCorpus = {
        "pin": "9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453",
        "frozen_blobs": verified,
        "source_rows": result.stdout.splitlines(),
    }
    output.write_text(json.dumps(corpus, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    regenerate(args.source, args.inventory, args.output)
