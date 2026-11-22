"""Capture complete frozen Python addition/subtraction binding dispatch."""

import argparse
import hashlib
import importlib
import json
import sys
from itertools import product
from pathlib import Path
from types import ModuleType
from typing import Any

from flagquantum.compiler.formats.kmx_block_marks import encode_kmx_block_marks
from flagquantum.compiler.registers.arithmetic.addition_binding import (
    physical_addition_binding,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient as Bit,
)
from tests.unit.test_addition_binding import _Case, _Kind, _Port, binding_ports


def source_records(source: ModuleType, case: _Case) -> str:
    n, m, kind, _, work, tolerance, subtract, _, _ = case
    target, offset, control, _, count, quantum = binding_ports(case)
    builder = source.CircuitBuilder()
    builder.create_quantum_register(quantum, name="q")
    builder.create_classical_register(count, name="b")
    builder.free(builder.alloc_qubits(work))
    builder.set_max_qubits(quantum + work)

    def port(item: _Port) -> Any:
        if isinstance(item, Bit):
            return source.b(item.slot)
        return source.q(item) if type(item) is int else item

    rhs = (
        offset if isinstance(offset, int) else [port(item) for item in reversed(offset)]
    )
    lhs = [source.q(item) for item in reversed(target)]
    append = builder.isub if subtract else builder.iadd
    for _ in range(2):
        append(rhs, target=lhs, control=port(control), btol=float(tolerance))
        assert builder.num_allocated_qubits == quantum
    return "\n".join(
        line
        for line in str(builder.finish_circuit()).splitlines()
        if not line.startswith(("APPEND_TO_REGISTER", "REGISTER"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    expected_blobs = {
        "iadd.pybind.cc": "919978108730d711dcd2465e29678fe238c0d3a6",
        "iadd.pybind.h": "42b6c225590a5ef0deb11fd40b85d39c92ce6e39",
        "iadd_test.py": "fc5a2f6731b3f5d81167725d578a03d45c0c99b0",
    }
    files = []
    for name, expected in expected_blobs.items():
        path = "kickmix/src/kickmix/py/circuit/subs/" + name
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
    refusals: list[_Case] = []
    kinds: tuple[_Kind, ...] = ("q", "b", "literal", "int")
    for n in (0, 1, 2, 3, 4, 5, 6):
        for m, kind, control, work, tolerance, subtract in product(
            sorted({0, 1, n, n + 1}),
            kinds,
            ("q", "b", "true", "false"),
            sorted({0, n, n + 2}),
            ("inf", "2.5", "nan", "-inf"),
            (False, True),
        ):
            if kind == "int" and m != n:
                continue
            case: _Case = (n, m, kind, control, work, tolerance, subtract, 0, "")
            target, offset, enable, clean, _, _ = binding_ports(case)
            try:
                original = source_records(source, case)
            except (ValueError, RuntimeError) as source_error:
                try:
                    physical_addition_binding(
                        offset,
                        target=target,
                        control=enable,
                        workspace=clean,
                        btol=float(tolerance),
                        subtract=subtract,
                    )
                except ValueError:
                    refusals.append(case)
                    continue
                raise AssertionError(
                    ("Native accepted source refusal", case)
                ) from source_error
            program, calls = physical_addition_binding(
                offset,
                target=target,
                control=enable,
                workspace=clean,
                btol=float(tolerance),
                subtract=subtract,
            )
            native = "\n".join(encode_kmx_block_marks(program, calls).splitlines()[1:])
            assert original == ("\n".join((native, native)) if native else ""), (
                case,
                original,
                native,
            )
            cases.append(
                (
                    *case[:7],
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
                "source_rejections": refusals,
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    print(f"Matched {len(cases)} repeated records and {len(refusals)} refusals")


if __name__ == "__main__":
    main()
