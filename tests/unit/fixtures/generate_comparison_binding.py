"""Capture complete dispatch records from the frozen Python extension.

The mixed, zero-clean-workspace SIGSEGV witness is documented in
COMPARISON_BINDING_AUDIT.md and excluded before entering the original C++ code.
"""

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
from flagquantum.compiler.registers.comparison.physical_comparison import (
    physical_comparison,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient as Bit,
)
from flagquantum.compiler.registers.primitives._mixed_gates import _XBasisTarget
from tests.unit.test_comparison_binding import _Case, _Port, _Relation, binding_ports


def _source_records(source: ModuleType, case: _Case) -> str:
    width, kind, target_kind, control_kind, work, tolerance, equal, relation, _, _ = (
        case
    )
    left, right, target, control, _, bit_count = binding_ports(
        width, kind, target_kind, control_kind, work
    )
    builder = source.CircuitBuilder()
    quantum_count = width + sum(type(port) is int for port in right) + 2
    builder.create_quantum_register(quantum_count, name="q")
    builder.create_classical_register(bit_count, name="b")
    builder.free(builder.alloc_qubits(work))
    builder.set_max_qubits(quantum_count + work)

    def port(value: _Port) -> Any:
        # External pybind objects remain local to this reference boundary.
        if isinstance(value, Bit):
            return source.b(value.slot)
        return source.q(value) if type(value) is int else value

    lhs = [source.q(wire) for wire in left[::-1]]
    rhs = (
        sum(1 << index for index, bit in enumerate(right[::-1]) if bit is True)
        if kind == "int"
        else [port(bit) for bit in right[::-1]]
    )
    if isinstance(target, _XBasisTarget):
        output = (
            source.xb(target.outcome.slot)
            if isinstance(target.outcome, Bit)
            else "|->" if target.outcome else "|+>"
        )
    else:
        output = "|->" if target is None else source.q(target)
    options = {"target": output, "control": port(control)}
    if relation != "eq":
        options.update(or_equal=equal, btol=float(tolerance))
    append = getattr(
        builder,
        {
            "lt": "flip_if_less_than",
            "gt": "flip_if_greater_than",
            "eq": "flip_if_equal",
        }[relation],
    )
    for _ in range(2):
        append(lhs, rhs, **options)
        assert builder.num_allocated_qubits == quantum_count
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
    metadata = json.loads(args.output.read_text())
    for item in metadata["source_files"]:
        data = source_root.parent.joinpath(item["path"]).read_bytes()
        assert (
            hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            == item["blob"]
        )
    sys.path.insert(0, str(source_root))
    source = importlib.import_module("kickmix")
    assert source.__version__ == "0.1" and hasattr(source, "Simulator")
    assert source.__file__ is not None
    assert Path(source.__file__).resolve().parent == source_root
    cases: list[_Case] = []
    rejections: list[_Case] = []
    crash_exclusions = 0
    relations: tuple[_Relation, ...] = ("lt", "gt", "eq")
    for width, kind, target, control, full, tolerance, equal, relation in product(
        (0, 1, 2, 4, 8),
        ("q", "b", "m", "int"),
        ("q", "phase", "c", "skip"),
        ("q", "true", "false", "c", "same"),
        (False, True),
        ("inf", "1.5"),
        (False, True),
        relations,
    ):
        if control == "same" and target != "c":
            continue
        if relation == "eq" and (tolerance != "inf" or equal):
            continue
        work = width if full else 0
        if (
            width >= 2
            and kind == "m"
            and not full
            and target != "skip"
            and control != "false"
            and tolerance == "inf"
            and relation != "eq"
        ):
            crash_exclusions += 1
            continue
        case: _Case = (
            width,
            kind,
            target,
            control,
            work,
            tolerance,
            equal,
            relation,
            0,
            "",
        )
        left, right, output, guard, clean, _ = binding_ports(
            width, kind, target, control, work
        )
        try:
            original = _source_records(source, case)
        except (ValueError, RuntimeError) as original_error:
            try:
                physical_comparison(
                    left,
                    right,
                    output,
                    clean,
                    relation=relation,
                    control=guard,
                    or_equal=equal,
                    btol=float(tolerance),
                )
            except ValueError:
                rejections.append(case)
                continue
            raise AssertionError(
                f"Native accepted original rejected case {case}"
            ) from original_error
        program, calls = physical_comparison(
            left,
            right,
            output,
            clean,
            relation=relation,
            control=guard,
            or_equal=equal,
            btol=float(tolerance),
        )
        native = "\n".join(encode_kmx_block_marks(program, calls).splitlines()[1:])
        assert original == ("\n".join((native, native)) if native else ""), case
        cases.append(
            (
                *case[:8],
                len(native.splitlines()),
                hashlib.sha256(native.encode()).hexdigest(),
            )
        )
    metadata.update(
        cases=cases,
        source_rejections=rejections,
        unsafe_source_cases_excluded=crash_exclusions,
    )
    args.output.write_text(json.dumps(metadata, separators=(",", ":")) + "\n")
    print(
        f"Matched {len(cases)} repeated binding records, {len(rejections)} rejections, {crash_exclusions} unsafe exclusions"
    )


if __name__ == "__main__":
    main()
