# SPDX-License-Identifier: Apache-2.0
# Observations from unchanged locked build/circuit_builder.cc and its header.

"""Compile the original result lifecycle and record its actual execution."""

import csv
import hashlib
import json
import re
import subprocess
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict

from tests.unit.fixtures.generate_physical_circuit_binding import observe

# The pybind oracle has no Python attribute stubs.
km = import_module("kickmix")

FTQC_SOURCE_INVENTORY = (
    Path(__file__).resolve().parents[3] / "docs/development/ftqc_source_inventory.csv"
)


class ResultObservation(TypedDict):
    columns: dict[str, list[int]]
    observation: dict[str, object]


def verify_sources(source: Path) -> None:
    inventory = FTQC_SOURCE_INVENTORY
    with inventory.open(encoding="utf-8-sig") as stream:
        expected = {row["path"]: row["git_blob"] for row in csv.DictReader(stream)}
    pending = [
        source / name
        for name in (
            "kickmix/build/circuit_builder.cc",
            "kickmix/mem/monotonic_arena.cc",
            "kickmix/mem/util.cc",
        )
    ]
    verified = set()
    while pending:
        path = pending.pop().resolve()
        if path in verified:
            continue
        raw = path.read_bytes()
        key = "kickmix/src/" + path.relative_to(source).as_posix()
        assert (
            hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            == expected[key]
        )
        verified.add(path)
        for name in re.findall(r'^#include "([^"]+)"', raw.decode(), re.M):
            local = path.parent / name
            pending.append(local if local.exists() else source / name)
    print("Verified", len(verified), "locked source blobs")


def main() -> None:
    assert km.__version__ == "0.1" and hasattr(km, "Simulator")
    module_file = km.__file__
    assert module_file is not None
    assert hashlib.sha256(Path(module_file).read_bytes()).hexdigest() == (
        "5e849075b49d895cca761fae864c8561b54362a6158fbfcabe7e9eb6e96d789e"
    )
    root = Path(__file__).resolve().parents[3]
    source = root.parent / "ftqc-reference-env/kickmix/src"
    verify_sources(source)
    with TemporaryDirectory(prefix="fq-raw-builder-") as temporary:
        binary = Path(temporary) / "results"
        subprocess.run(
            [
                "clang++",
                "-std=c++20",
                "-O1",
                "-ffunction-sections",
                "-fdata-sections",
                "-Wl,-dead_strip",
                "-I" + str(source),
                str(Path(__file__).with_name("raw_builder_results.cc")),
                *(
                    str(source / name)
                    for name in (
                        "kickmix/build/circuit_builder.cc",
                        "kickmix/mem/monotonic_arena.cc",
                        "kickmix/mem/util.cc",
                    )
                ),
                "-o",
                str(binary),
            ],
            check=True,
            timeout=100,
            capture_output=True,
        )
        result = subprocess.run(
            [str(binary)], check=True, capture_output=True, text=True, timeout=10
        )
    lines = result.stdout.splitlines()
    names = {
        65: "BIT_STORE1",
        64: "BIT_STORE0",
        145: "X",
        80: "HMR",
        129: "PUSH_CONDITION",
        16: "Z",
        1: "POP_CONDITION",
        71: "DEBUG_PRINT",
    }
    rows: dict[str, ResultObservation] = {}
    for start in range(0, len(lines), 7):
        columns = {
            line.split()[0]: list(map(int, line.split()[1:]))
            for line in lines[start + 1 : start + 7]
        }
        bits = iter(columns["b0"])
        conditions = iter(columns["bc"])
        qubits = iter(columns["q0"])
        body = []
        for code in columns["ops"]:
            line = names[code]
            if code & 0x30:
                line += " q" + str(next(qubits))
            if code & 0x40:
                line += " b" + str(next(bits))
            if code & 0x80:
                line += " if b" + str(next(conditions))
            body.append(line)
        rows[lines[start]] = {
            "columns": columns,
            "observation": observe(km.Circuit("\n".join(body))),
        }
    destination = Path(__file__).parents[1] / "data/physical_builder_results.json"
    destination.write_text(json.dumps(rows, indent=2) + "\n")
    print("Recorded", len(rows), "complete original states with three executed lanes")


if __name__ == "__main__":
    main()
