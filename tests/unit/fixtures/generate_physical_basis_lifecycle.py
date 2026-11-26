"""Capture unmodified physical simulator lifecycle and ownership."""

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
        row["path"]: row["git_blob"]
        for row in csv.DictReader(
            inventory.read_text(encoding="utf-8-sig").splitlines()
        )
    }
    implementations = [
        path
        for folder in ("id", "circuit", "mem")
        for path in sorted((src / "kickmix" / folder).glob("*.cc"))
        if not path.name.endswith((".test.cc", ".perf.cc"))
    ]
    implementations.extend(
        src / "kickmix" / name
        for name in (
            "build/circuit_builder.cc",
            "build/circuit_pattern.cc",
            "py/circuit/ops/cx.cc",
            "py/circuit/ops/cz.cc",
            "py/circuit/ops/swap.cc",
            "sim/fuzzer.cc",
            "sim/sim.cc",
            "util/fixed_width_int.cc",
            "util/fixed_precision_angle_128.cc",
            "util/binary_file_tools.cc",
            "util/mod_rand.cc",
        )
    )
    pending = [
        *implementations,
        src / "kickmix/sim/fuzzer.test.cc",
        src / "kickmix/sim/sim.test.cc",
        src / "kickmix/sim/sim.perf.cc",
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
            raise ValueError(f"Frozen simulator source differs for {relative}")
        verified[relative] = blob
        for include in re.findall(r'^#include "([^"]+)"', raw.decode(), re.M):
            if include.startswith("gtest/") or include == "immintrin.h":
                continue
            local = path.parent / include
            pending.append(local if local.exists() else src / include)
    workloads = re.findall(
        r'R"CIRCUIT\((.*?)\)CIRCUIT"',
        (src / "kickmix/sim/sim.perf.cc").read_text(),
        re.S,
    )
    if len(workloads) != 3:
        raise ValueError("Expected the complete three original simulator workloads")
    with tempfile.TemporaryDirectory(prefix="fq-lifecycle-source-") as folder:
        binary = Path(folder) / "probe"
        subprocess.run(
            [
                "clang++",
                "-std=c++20",
                "-O1",
                "-I" + str(src),
                *map(str, implementations),
                str(Path(__file__).with_name("physical_basis_lifecycle.cc")),
                "-o",
                str(binary),
            ],
            check=True,
            capture_output=True,
            timeout=240,
        )
        result = subprocess.run(
            [str(binary)], check=True, text=True, capture_output=True, timeout=60
        )
        workload_results = [
            subprocess.run(
                [str(binary), str(mode)],
                input=text,
                check=True,
                text=True,
                capture_output=True,
                timeout=60,
            ).stdout
            for mode, text in enumerate(workloads)
        ]
    observations = result.stdout
    output.write_text(
        json.dumps(
            {
                "pin": "9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453",
                "frozen_blobs": verified,
                "observations": observations,
                "workloads": workloads,
                "workload_observations": workload_results,
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
