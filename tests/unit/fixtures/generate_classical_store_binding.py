"""Record every classical store overload through the frozen Python binding."""

import argparse
import hashlib
import importlib
import json
import sys
from itertools import product
from pathlib import Path

from tests.unit.fixtures.generate_pauli_broadcast_binding import _source_port
from tests.unit.fixtures.kickmix_binary_gate_matrix import Port


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.source_root.resolve()
    sys.path.insert(0, str(root))
    source = importlib.import_module("kickmix")
    assert source.__version__ == "0.1" and hasattr(source, "Simulator")
    assert source.__file__ is not None and Path(source.__file__).parent == root
    intake = json.loads(
        root.parent.joinpath("g2-bit-store-binding-intake-20261122.json").read_text()
    )
    for item in intake["source_files"]:
        data = root.parent.joinpath(item["path"]).read_bytes()
        assert (
            hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            == item["blob"]
        )
    targets: list[Port] = [
        "b0",
        "b1",
        "b3",
        "q0",
        "0",
        "xb0",
        "-",
        [],
        ["b0"],
        ["b0", "b1"],
        ["b1", "b0"],
        ["b0", "b0"],
        ["b0", "b1", "b0"],
        ["b3", "b1", "b0"],
        ["b2", "b3"],
        ["b0", "q0"],
    ]
    controls: list[Port] = [
        "1",
        "0",
        "b0",
        "b1",
        "b3",
        "q0",
        "xb0",
        "+",
        [],
        ["1"],
        ["b0"],
        ["1", "0"],
        ["b0", "b1"],
        ["b1", "b0"],
        ["b0", "1"],
        ["0", "b0"],
        ["b0", "q0"],
        ["b1", "b2", "b3"],
    ]
    cases: list[tuple[Port, object, Port, tuple[int, str] | None]] = []
    for target, value, control in product(targets, (False, True), controls):
        builder = source.CircuitBuilder()
        try:
            builder.bit_store(
                _source_port(source, target),
                value,
                control=_source_port(source, control),
            )
        except (TypeError, ValueError):
            assert not str(builder.finish_circuit()).strip()
            cases.append((target, value, control, None))
            continue
        records = str(builder.finish_circuit()).strip()
        cases.append(
            (
                target,
                value,
                control,
                (
                    len(records.splitlines()),
                    hashlib.sha256(records.encode()).hexdigest(),
                ),
            )
        )
    for literal in (0, 1, 2, -1, 0.0, 0.25, None, [], [1], "", "false"):
        builder = source.CircuitBuilder()
        try:
            builder.bit_store(source.b(0), literal)
        except (TypeError, ValueError, RuntimeError):
            assert not str(builder.finish_circuit()).strip()
            cases.append(("b0", literal, "1", None))
            continue
        records = str(builder.finish_circuit()).strip()
        cases.append(
            (
                "b0",
                literal,
                "1",
                (
                    len(records.splitlines()),
                    hashlib.sha256(records.encode()).hexdigest(),
                ),
            )
        )
    args.output.write_text(
        json.dumps({"source_files": intake["source_files"], "cases": cases}, indent=2)
        + "\n"
    )
    print(f"Captured {len(cases)} source store configurations")


if __name__ == "__main__":
    main()
