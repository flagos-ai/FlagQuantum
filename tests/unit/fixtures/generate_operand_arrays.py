"""Record unchanged locked operand array definitions against verified sources."""

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import TypedDict


class ArrayConsumer(TypedDict):
    operation: str
    enabled: bool
    kmx: str
    quantum_words: list[int]
    phase_half_turns: list[str]
    lane_counts: list[dict[str, int]]


def regenerate(source: Path, inventory: Path, output: Path) -> None:
    expected = {
        row["path"]: row["git_blob"]
        for row in csv.DictReader(
            inventory.read_text(encoding="utf-8-sig").splitlines()
        )
    }
    roots = [
        "array_z.h",
        "array_z.cc",
        "array_z.test.cc",
        "array_xz.h",
        "array_xz.cc",
        "stride_span_x.h",
        "stride_span_xz.h",
        "stride_span_xz.cc",
        "stride_span_xz.test.cc",
        "stride_span_z.h",
        "stride_span_z.cc",
        "stride_span_z.test.cc",
    ]
    src = (source / "kickmix/src").resolve()
    implementations = sorted(
        p
        for p in (src / "kickmix/id").glob("*.cc")
        if not p.name.endswith((".test.cc", ".perf.cc"))
    ) + [src / "kickmix/mem/util.cc", src / "kickmix/util/fixed_width_int.cc"]
    pending = [src / "kickmix/id" / name for name in roots] + implementations
    verified = {}
    while pending:
        path = pending.pop().resolve()
        relative = "kickmix/src/" + path.relative_to(src).as_posix()
        if relative in verified:
            continue
        raw = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if blob != expected[relative]:
            raise ValueError(f"Frozen operand array source differs for {relative}")
        verified[relative] = blob
        for include in re.findall(r'^#include "([^"]+)"', raw.decode(), re.M):
            if not include.startswith("gtest/"):
                pending.append(
                    src / include
                    if include.startswith("kickmix/")
                    else path.parent / include
                )
    with tempfile.TemporaryDirectory(prefix="fq-operand-array-oracle-") as folder:
        binary = Path(folder) / "oracle"
        command = [
            "clang++",
            "-std=c++20",
            "-O1",
            "-I" + str(src),
            str(Path(__file__).with_name("operand_arrays.cc")),
            *map(str, implementations),
            "-o",
            str(binary),
        ]
        subprocess.run(command, check=True, capture_output=True, timeout=120)
        rows = subprocess.run(
            [str(binary)], check=True, text=True, capture_output=True, timeout=15
        ).stdout.splitlines()
    km = importlib.import_module("kickmix")
    extension = Path(inspect.getfile(km.Circuit))
    if (
        km.__version__ != "0.1"
        or not hasattr(km, "Simulator")
        or extension.suffix != ".so"
    ):
        raise RuntimeError(
            "The locked native Kickmix extension and Simulator are required"
        )
    consumers: list[ArrayConsumer] = []
    for operation in (
        "identity",
        "reversed",
        "skip",
        "keep",
        "skip_last",
        "keep_last",
        "step2",
        "repeat",
    ):
        for enabled in (False, True):
            controls = km.array([km.q(0), km.b(0), True, False])
            targets = km.array([km.q(1), km.q(2), km.q(3), km.q(4)])
            if operation == "reversed":
                controls, targets = controls[::-1], targets[::-1]
            elif operation == "step2":
                controls, targets = controls[::2], targets[::2]
            elif operation == "skip":
                controls, targets = controls[1:], targets[1:]
            elif operation == "keep":
                controls, targets = controls[:2], targets[:2]
            elif operation == "skip_last":
                controls, targets = controls[:-1], targets[:-1]
            elif operation == "keep_last":
                controls, targets = controls[-2:], targets[-2:]
            elif operation == "repeat":
                controls = km.array([False] * 4)
            builder = km.CircuitBuilder()
            builder.cx(controls, targets)
            circuit = builder.finish_circuit()
            simulator = km.Simulator(
                batch_size=32, counting_operations=True, debug_print=False, seed=1
            )
            for bit in range(5):
                simulator.write_across_shots(
                    km.q(bit),
                    sum(((lane >> (4 - bit)) & 1) << lane for lane in range(32)),
                )
            simulator.write_across_shots(km.b(0), ((1 << 32) - 1) * enabled)
            simulator.do(circuit)
            consumers.append(
                {
                    "operation": operation,
                    "enabled": enabled,
                    "kmx": str(circuit),
                    "quantum_words": [
                        int(simulator.read_across_shots(km.q(k), out=int))
                        for k in range(5)
                    ],
                    "phase_half_turns": [
                        str(simulator.read_shot_phase(k)) for k in range(32)
                    ],
                    "lane_counts": [
                        dict(simulator.operation_statistics(k).items())
                        for k in range(32)
                    ],
                }
            )
    output.write_text(
        json.dumps(
            {
                "pin": "9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453",
                "frozen_blobs": verified,
                "roots": roots,
                "source_rows": rows,
                "gate_consumers": consumers,
                "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
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
