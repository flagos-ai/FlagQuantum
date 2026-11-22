"""Record complete lookup Python calls from the unchanged native extension."""

import argparse
import hashlib
import importlib
import json
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import TypedDict


class Case(TypedDict):
    address_width: int
    target_width: int
    route: str
    table: list[int | float | str]
    classical_width: int
    reuse_pool: bool


def source_records(source: ModuleType, case: Case) -> str:
    builder = source.CircuitBuilder()
    n, w = case["address_width"], case["target_width"]
    address = builder.create_quantum_register(n, name="address")
    target = builder.create_quantum_register(w, name="target")
    if case["classical_width"]:
        builder.create_classical_register(case["classical_width"], name="bits")
    if case["reuse_pool"] and n:
        builder.free(builder.alloc_qubits(n))
    for erase in (
        (False, True) if case["route"] == "joined" else (case["route"] == "erase",)
    ):
        getattr(builder, "del_lookup" if erase else "init_lookup")(
            table=case["table"], address=address, target=target
        )
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
    path = Path(source.__file__).parent / "src/kickmix/py/circuit/subs/lookup.pybind.cc"
    raw = path.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    assert blob == "227392313663d4c2c06987e47d93c226869bf4be"
    from flagquantum.compiler.formats.kmx_block_marks import encode_kmx_block_marks
    from tests.unit.test_lookup_binding import native_trace

    cases = []
    for n in range(7):
        for w in range(5):
            for route in ("init", "erase", "joined"):
                if n == 6 and w == 1 and route != "init":
                    continue  # The original wrapper underallocates erasure work.
                for mode in range(5):
                    for declared, reuse in ((0, False), (3, True)):
                        table: list[int | float | str] = [
                            (
                                0
                                if mode == 0
                                else (
                                    (1 << w) - 1
                                    if mode == 1
                                    else (5 * row + 1) % (1 << w)
                                )
                            )
                            for row in range(1 << n)
                        ]
                        if mode == 3:
                            table = [str(value) for value in table]
                        elif mode == 4:
                            table = [int(value) + 0.5 for value in table]
                        case = Case(
                            address_width=n,
                            target_width=w,
                            route=route,
                            table=table,
                            classical_width=declared,
                            reuse_pool=reuse,
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
                                    )[:60]
                                )
                            )
                            raise AssertionError("Complete lookup binding differs")
                        cases.append(
                            {
                                "input": case,
                                "records": len(original.splitlines()),
                                "counts": dict(
                                    Counter(
                                        line.split()[0]
                                        for line in original.splitlines()
                                    )
                                ),
                                "sha256": hashlib.sha256(original.encode()).hexdigest(),
                            }
                        )
    args.output.write_text(
        '{\n  "source_blob": '
        + json.dumps(blob)
        + ',\n  "cases": [\n'
        + ",\n".join("    " + json.dumps(row, separators=(",", ":")) for row in cases)
        + "\n  ]\n}\n"
    )
    print(f"Matched {len(cases)} complete original lookup streams")


if __name__ == "__main__":
    main()
