"""Record the frozen Python field constructor and scalar binding observations."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any


@dataclass
class IndexValue:
    value: int

    def __index__(self) -> int:
        return self.value

    def __int__(self) -> int:
        return self.value


class IntegerSubclass(int):
    pass


def external_value(value: Any) -> Any:
    """Materialize JSON-only object descriptions independently of native parsing."""
    if not isinstance(value, dict):
        return value
    kind = value["kind"]
    if kind == "index":
        return IndexValue(value["value"])
    if kind == "subclass":
        return IntegerSubclass(value["value"])
    if kind == "poly":

        class Polynomial:
            field = type("CoefficientField", (), {"characteristic": value["char"]})
            degree = value.get("degree", 8)
            coeffs = (1,)

            def __int__(self) -> int:
                return int(value["value"])

        return Polynomial()
    if kind == "element":

        class Element(IndexValue):
            characteristic = value["char"]

        return Element(value["value"])
    if kind == "field":
        attributes = {
            "characteristic": value["char"],
            "degree": value["degree"],
            "irreducible_poly": external_value(value["modulus"]),
        }
        if "primitive" in value:
            attributes["primitive_element"] = external_value(value["primitive"])
        return type("ExternalField", (), attributes)
    raise ValueError(f"Unknown fixture object kind {kind}")


def requests() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    moduli: list[Any] = [
        None,
        0x11B,
        0x1B,
        "x^8+x^4+x^3+x+1",
        "0x11B",
        "0x",
        True,
        -1,
        1 << 1024,
        0x101,
        8.0,
        {"kind": "index", "value": 0x11B},
        {"kind": "poly", "char": 2, "value": 0x11B},
        {"kind": "poly", "char": 3, "value": 0x11B},
        {"kind": "element", "char": 3, "value": 0x11B},
        {"kind": "element", "char": 2, "value": 0x11B},
    ]
    for degree in (None, 8, True, "8", -1, 0, 513, 1 << 100):
        for modulus in moduli:
            cases.append({"kind": "construct", "args": [degree, modulus, None]})
    for primitive in (
        None,
        0,
        1,
        2,
        3,
        5,
        255,
        256,
        True,
        -1,
        {"kind": "index", "value": 3},
        {"kind": "poly", "char": 2, "value": 3},
    ):
        cases.append({"kind": "construct", "args": [8, 0x11B, primitive]})
    for characteristic in (2, 3, True, 2.0):
        for external_degree in (8, True, 8.0, 513):
            field = {
                "kind": "field",
                "char": characteristic,
                "degree": external_degree,
                "modulus": {"kind": "poly", "char": 2, "value": 0x11B},
            }
            for modulus in (None, 0x11B):
                cases.append({"kind": "construct", "args": [field, modulus, None]})
    cases.append(
        {"kind": "construct", "args": [{"kind": "subclass", "value": 8}, 0x11B, None]}
    )
    for width in (1, 2, 4, 8, 65, 128, 512):
        values = (0, 1, 2, (1 << width) - 1, (1 << (width + 3)) | 7)
        for left in values:
            for right in values:
                cases.append({"kind": "add", "degree": width, "args": [left, right]})
            for exponent in (0, 1, -1, -2, 255, (1 << 160) | 17, -(1 << 160) - 17):
                cases.append({"kind": "pow", "degree": width, "args": [left, exponent]})
    for kind, args in (
        ("add", [True, 1]),
        ("pow", [True, -1]),
        ("pow", [2, True]),
        ("add", [-1, 0]),
        ("add", [1 << 1024, 0]),
        ("pow", [2, 1.5]),
        ("pow", [2, "3"]),
        ("pow", [2, {"kind": "index", "value": 2}]),
    ):
        cases.append({"kind": kind, "degree": 4, "args": args})
    for count in (
        0,
        1,
        True,
        -1,
        (1 << 64) - 1,
        1 << 64,
        1.0,
        "2",
        {"kind": "index", "value": 2},
    ):
        cases.append({"kind": "frobenius", "degree": 4, "args": [3, count]})
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = import_module("kickmix")
    assert reference.__version__ == "0.1" and hasattr(reference, "Simulator")
    cases = requests()
    for case in cases:
        operands = [external_value(value) for value in case["args"]]
        try:
            if case["kind"] == "construct":
                field = reference.GF2Field(*operands)
                result = [field.degree, field.modulus, field.primitive_element]
            else:
                field = reference.GF2Field(case["degree"])
                result = getattr(field, case["kind"])(*operands)
            case["result"] = result
        except (ValueError, TypeError) as error:
            case["error"] = type(error).__name__
    assert reference.__file__ is not None
    source = (
        Path(reference.__file__).parent
        / "src/kickmix/py/circuit/subs/gf_arithmetic.pybind.cc"
    )
    raw = source.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    assert blob == "5f111727a1e383394ef856e7104ff4817ae12f31"
    args.output.write_text(
        json.dumps({"source_blob": blob, "cases": cases}, indent=2) + "\n"
    )
    print(f"Recorded {len(cases)} complete original Python binding observations")


if __name__ == "__main__":
    main()
