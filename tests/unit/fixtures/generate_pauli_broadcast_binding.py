"""Capture the frozen unary binding matrix and original exact-phase consumers."""

import argparse
import ast
import hashlib
import importlib
import json
import sys
from fractions import Fraction
from itertools import product
from pathlib import Path
from types import ModuleType
from typing import Any

from tests.unit.fixtures.kickmix_binary_gate_matrix import Port
from tests.unit.fixtures.kickmix_source_oracle import normalized_stream


def _source_port(source: ModuleType, value: Port) -> Any:
    if isinstance(value, list):
        return source.array([_source_port(source, item) for item in value])
    if value in ("0", "1"):
        return value == "1"
    if value in ("+", "-"):
        return source.xbool(value == "-")
    if value.startswith("xb"):
        return source.xb(int(value[2:]))
    return getattr(source, value[0])(int(value[1:]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    sys.path.insert(0, str(root))
    source = importlib.import_module("kickmix")
    assert source.__version__ == "0.1" and hasattr(source, "Simulator")
    assert source.__file__ is not None and Path(source.__file__).parent == root
    intake = root.parent.joinpath("g2-pauli-binding-intake-20261122.json")
    files = json.loads(intake.read_text())
    for item in files:
        data = root.parent.joinpath(item["path"]).read_bytes()
        assert (
            hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            == item["blob"]
        )
    tokens = ("q0", "q2", "b0", "b2", "xb0", "xb2", "+", "-", "0", "1")
    ports: list[Port] = [*tokens, [], *map(list, product(tokens, repeat=2))]
    ports.extend([[f"{kind}{i}" for i in (4, 2, 0)] for kind in ("q", "b", "xb")])
    ports.extend([["-", "-", "+", "-"], ["1", "0", "1", "1"]])
    cases: list[tuple[str, Port, tuple[int, str] | None]] = []
    for opcode, port in product(("x", "z"), ports):
        builder = source.CircuitBuilder()
        try:
            getattr(builder, opcode)(_source_port(source, port))
        except (TypeError, ValueError):
            assert not str(builder.finish_circuit()).strip()
            cases.append((opcode, port, None))
            continue
        text = str(builder.finish_circuit())
        records = normalized_stream(text)
        cases.append(
            (
                opcode,
                port,
                (len(records), hashlib.sha256(repr(records).encode()).hexdigest()),
            )
        )
    builder = source.CircuitBuilder()
    for wire, angle in enumerate((0.25, Fraction(1, 10), 0.1)):
        builder.z_pow(source.q(wire), angle)
    header = 'INFO_TAG "@flagquantum:reversible-phase-v1 wires=3 bits=0"\n'
    simple = header + str(builder.finish_circuit()) + "\n"
    module = ast.parse(
        root.joinpath("src/kickmix/py/circuit/ops/z_pow_test.py").read_text()
    )
    (function,) = (
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "test_ccz_decomposed_into_7_z_pow"
    )
    (ccz_text,) = (
        node.args[0].value
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Circuit"
        and isinstance(node.args[0], ast.Constant)
    )
    ccz = header + str(source.Circuit(ccz_text)) + "\n"
    args.output.write_text(
        json.dumps(
            {
                "source_files": files,
                "cases": cases,
                "zpow_simple": simple,
                "zpow_ccz": ccz,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Captured {len(cases)} original binding cases and both exact-phase consumers"
    )


if __name__ == "__main__":
    main()
