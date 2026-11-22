"""Compare complete frozen GF binding calls with native physical composition."""

import argparse
import hashlib
import importlib
import json
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, TypedDict

from flagquantum.compiler.formats.kmx_block_marks import encode_kmx_block_marks
from flagquantum.compiler.registers.finite_field.division_chain import (
    field_division_workspace_size,
)
from flagquantum.compiler.registers.finite_field.inverse_chain import (
    field_inverse_chain_size,
    field_inverse_workspace_size,
)


class Case(TypedDict):
    width: int
    route: str
    classical_width: int
    reuse_pool: bool
    allocate_target: bool
    modulus: int | None


def work_size(case: Case) -> int:
    width, route = case["width"], case["route"]
    if "div" in route and route not in ("constant_div",):
        return field_division_workspace_size(width)
    if "inverse" in route:
        return (
            field_inverse_chain_size
            if "scaffold" in route
            else field_inverse_workspace_size
        )(width)
    return width if "product" in route and "quantum" in route else 0


def source_records(source: ModuleType, case: Case) -> str:
    width, route = case["width"], case["route"]
    builder = source.CircuitBuilder()
    left = builder.create_quantum_register(width, name="left")
    right = builder.create_quantum_register(width, name="right")
    control = builder.create_quantum_register(1, name="control")
    output = (
        None
        if case["allocate_target"]
        else builder.create_quantum_register(width, name="target")
    )
    count = max(
        case["classical_width"],
        width if route == "phase" else 1 if "classical" in route else 0,
    )
    bits: Any = builder.create_classical_register(count, name="bits") if count else None
    clean = work_size(case)
    base = 2 * width + 1 + (0 if case["allocate_target"] else width)
    if case["reuse_pool"] and clean:
        builder.free(builder.alloc_qubits(clean))
    builder.set_max_qubits(base + clean + (width if case["allocate_target"] else 0))
    field = None if case["modulus"] is None else source.GF2Field(width, case["modulus"])
    target: Any = "alloc" if output is None else output
    if route.startswith("add"):
        offset = 3 if "constant" in route else bits[:width] if "bits" in route else left
        enable = (
            control[0]
            if "quantum" in route
            else (
                bits[count - 1]
                if "classical" in route
                else False if "false" in route else True
            )
        )
        builder.gf2_iadd(offset, target=target, control=enable, field=field)
    elif route.startswith("constant"):
        getattr(builder, "gf2_idiv" if route.endswith("div") else "gf2_imul")(
            1 if width == 1 else 3, target=target, field=field
        )
    elif route == "phase":
        builder.gf2_phase_by_product(left, right, mask=bits[:width], field=field)
    elif "product" in route:
        mode = (
            "init"
            if route.startswith("init")
            else "del" if route.startswith("erase") else "ixor"
        )
        enable = (
            control[0]
            if "quantum" in route
            else (
                bits[count - 1]
                if "classical" in route
                else False if "false" in route else True
            )
        )
        getattr(builder, mode + "_gf2_mul")(
            left, right, target=target, field=field, control=enable
        )
    elif "inverse" in route:
        if route.startswith("erase"):
            if "scaffold" in route:
                scaffold = builder.alloc_qubits(clean)
                builder.del_gf2_inverse_with_scaffold(
                    left, target=target, scaffold=scaffold, field=field
                )
            else:
                builder.del_gf2_inverse(left, target=target, field=field)
        else:
            getattr(
                builder,
                "init_gf2_inverse" + ("_with_scaffold" if "scaffold" in route else ""),
            )(left, target=target, field=field)
    else:
        rhs = left if route.endswith("self") else right
        if "scaffold" in route and route.startswith("erase"):
            scaffold = builder.alloc_qubits(clean)
            builder.del_gf2_div_with_scaffold(
                left, rhs, target=target, scaffold=scaffold, field=field
            )
        else:
            prefix = (
                "init"
                if route.startswith("init")
                else "del" if route.startswith("erase") else "ixor"
            )
            getattr(
                builder,
                prefix + "_gf2_div" + ("_with_scaffold" if "scaffold" in route else ""),
            )(left, rhs, target=target, field=field)
    return "\n".join(
        line
        for line in str(builder.finish_circuit()).splitlines()
        if not line.startswith(("REGISTER", "APPEND_TO_REGISTER"))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = importlib.import_module("kickmix")
    assert source.__version__ == "0.1" and hasattr(source, "Simulator")
    assert source.__file__ is not None
    source_path = (
        Path(source.__file__).parent
        / "src/kickmix/py/circuit/subs/gf_arithmetic.pybind.cc"
    )
    raw = source_path.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    assert blob == "5f111727a1e383394ef856e7104ff4817ae12f31"
    from tests.unit.test_field_physical_binding import native_trace

    routes = (
        "add",
        "add_constant",
        "add_quantum",
        "add_classical",
        "add_bits",
        "add_false",
        "constant_mul",
        "constant_div",
        "phase",
        "init_product",
        "xor_product",
        "erase_product",
        "init_product_quantum",
        "xor_product_quantum",
        "erase_product_quantum",
        "init_product_classical",
        "erase_product_classical",
        "init_product_false",
        "xor_product_false",
        "init_inverse",
        "erase_inverse",
        "init_inverse_scaffold",
        "erase_inverse_scaffold",
        "init_div",
        "xor_div",
        "erase_div",
        "init_div_scaffold",
        "erase_div_scaffold",
        "xor_div_self",
    )
    cases = []
    for width in (1, 2, 3, 4, 8, 17):
        for route in routes:
            if width == 1 and route == "add_constant":
                continue
            for declared, reuse in ((0, False), (max(width, 3), True)):
                if route == "add_bits":
                    declared = max(declared, width)
                for allocated in (
                    (False, True) if route.startswith("init") else (False,)
                ):
                    # Reused target storage and scaffold allocation are compared in a separate whole-pool case.
                    if allocated and reuse:
                        continue
                    case = Case(
                        width=width,
                        route=route,
                        classical_width=declared,
                        reuse_pool=reuse,
                        allocate_target=allocated,
                        modulus=0x11B if width == 8 and reuse else None,
                    )
                    original = source_records(source, case)
                    native = "\n".join(
                        encode_kmx_block_marks(*native_trace(case)).splitlines()[1:]
                    )
                    if original != native:
                        import difflib

                        print(case)
                        print(
                            "\n".join(
                                list(
                                    difflib.unified_diff(
                                        original.splitlines(), native.splitlines()
                                    )
                                )[:70]
                            )
                        )
                        raise AssertionError("Complete GF binding stream differs")
                    cases.append(
                        {
                            "input": case,
                            "records": len(original.splitlines()),
                            "counts": dict(
                                Counter(
                                    line.split()[0] for line in original.splitlines()
                                )
                            ),
                            "sha256": hashlib.sha256(original.encode()).hexdigest(),
                        }
                    )
    args.output.write_text(
        json.dumps({"source_blob": blob, "cases": cases}, indent=2) + "\n"
    )
    print(f"Matched {len(cases)} complete original GF binding streams")


if __name__ == "__main__":
    main()
