"""Verify complete locked multi-control binding and repeated LIFO leases."""

import argparse
import hashlib
import importlib
import json
import sys
from itertools import product
from pathlib import Path
from types import ModuleType
from typing import Any

from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.registers.control.physical_multi_control import (
    physical_multi_controlled_x,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient as Bit,
)
from tests.unit.test_multi_control_binding import _Case, _Pattern, _Port, binding_ports


def source_records(source: ModuleType, case: _Case) -> str:
    controls, targets, _, quantum = binding_ports(case)
    builder = source.CircuitBuilder()
    builder.create_quantum_register(quantum, name="q")
    builder.create_classical_register(
        max((item.slot for item in controls if isinstance(item, Bit)), default=-1) + 1,
        name="b",
    )
    builder.free(builder.alloc_qubits(case[3]))
    builder.set_max_qubits(quantum + case[3])

    def port(item: _Port) -> Any:
        if isinstance(item, Bit):
            return source.b(item.slot)
        return source.q(item) if type(item) is int else item

    for _ in range(2):
        builder.multi_controlled_x(
            [port(item) for item in controls], [source.q(item) for item in targets]
        )
        assert builder.num_allocated_qubits == quantum
    return "\n".join(
        line
        for line in str(builder.finish_circuit()).splitlines()
        if not line.startswith(("REGISTER", "APPEND_TO_REGISTER"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    expected_blobs = {
        "kickmix/src/kickmix/py/circuit/subs/multi_control_x.pybind.cc": "1ed2498237a0f3ae76673f2ddef57ffc3e91b374",
        "kickmix/src/kickmix/py/circuit/subs/multi_control_x.pybind.h": "ada09a99b20d649558cadecc75767913f3a2b819",
        "kickmix/src/kickmix/py/circuit/subs/multi_control_test.py": "c7b3b34b2b245190622f55728af8b4675c80bfb5",
        "kickmix/src/kickmix/build/circuit_pattern.cc": "46a69f4144b1cb5c41d8800a31175b8d83ea10fa",
        "kickmix/src/kickmix/build/circuit_pattern.h": "aaae344e260bef14a204fc34ec2183b4e4e2a5e7",
    }
    files = []
    for path, expected in expected_blobs.items():
        data = source_root.parent.joinpath(path).read_bytes()
        actual = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data
        ).hexdigest()
        assert actual == expected, path
        files.append({"path": path, "blob": actual})
    sys.path.insert(0, str(source_root))
    source = importlib.import_module("kickmix")
    assert source.__version__ == "0.1" and hasattr(source, "Simulator")
    assert source.__file__ and Path(source.__file__).resolve().parent == source_root
    cases: list[_Case] = []
    patterns: tuple[_Pattern, ...] = ("q", "mixed", "duplicate", "false")
    for n, m, pattern, extra in product(
        (*range(7), 10, 50), (0, 1, 2, 4), patterns, (0, 2)
    ):
        work = max(0, n - 2) + extra
        case: _Case = (n, m, pattern, work, 0, "")
        controls, target, leases, _ = binding_ports(case)
        used = max(0, n - 2) if m and pattern != "false" else 0
        returned = (*reversed(leases[:used]), *leases[used:])
        original = source_records(source, case)
        streams = [
            "\n".join(
                encode_kmx(
                    physical_multi_controlled_x(controls, target, workspace=clean)
                ).splitlines()[1:]
            )
            for clean in (leases, returned)
        ]
        native = "\n".join(stream for stream in streams if stream)
        assert original == native, (case, original, native)
        cases.append(
            (
                *case[:4],
                len(native.splitlines()),
                hashlib.sha256(native.encode()).hexdigest(),
            )
        )
    args.output.write_text(
        json.dumps(
            {
                "source_revision": "9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453",
                "source_files": files,
                "cases": cases,
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    print(f"Matched {len(cases)} complete repeated-call binding records")


if __name__ == "__main__":
    main()
