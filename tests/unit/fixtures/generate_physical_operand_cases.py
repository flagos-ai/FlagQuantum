# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Record every locked operation-test helper family without changing the source."""

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import kickmix as km


def main() -> None:
    assert km.__version__ == "0.1" and hasattr(km, "Simulator")
    assert hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest() == (
        "5e849075b49d895cca761fae864c8561b54362a6158fbfcabe7e9eb6e96d789e"
    )
    source = Path(km.__file__).parent / "src/kickmix/py/circuit/ops/op_testing_util.py"
    raw = source.read_bytes()
    assert (
        hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        == "8fcef09055bfe6d6d7e68eb3f6450ad1f60299bb"
    )
    spec = importlib.util.spec_from_file_location("frozen_op_testing", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def encode(value: Any) -> list[object]:
        if isinstance(value, bool):
            return ["bool", int(value)]
        for kind, cls in (("q", km.q), ("b", km.b), ("xb", km.xb), ("xbool", km.xbool)):
            if isinstance(value, cls):
                return [kind, int(bool(value)) if kind == "xbool" else value.id]
        return [
            "array" if isinstance(value, km.array) else "list",
            [encode(item) for item in value],
        ]

    cases = []
    for index in (0, 1, 5):
        for size in (-1, 0, 1, 3, 8):
            for include in (False, True):
                cases.append(
                    {
                        "index": index,
                        "size": size,
                        "include_x_bits": include,
                        "x": [
                            encode(v)
                            for v in module.x_list_cases(
                                index, size, include_x_bits=include
                            )
                        ],
                        "z": [encode(v) for v in module.z_list_cases(index, size)],
                        "scalar_xq": [encode(v) for v in module.x_cases_q(index)],
                        "scalar_xbc": [encode(v) for v in module.x_cases_bc(index)],
                        "scalar_z": [encode(v) for v in module.z_cases(index)],
                    }
                )
    constants = {
        name: getattr(module, name)
        for name in ("U64_FALSE", "U64_TRUE", *[f"U64_BIT{k}" for k in range(6)])
    }
    broadcasts = []
    for size in (0, 1, 3):
        for literal in (False, True):
            args = (km.q(5), [km.b(k) for k in range(size)], literal, km.xbool(literal))
            broadcasts.append(
                {
                    "size": size,
                    "literal": literal,
                    "result": encode(module.broadcast_logic(*args)),
                }
            )
    broadcasts.append(
        {
            "args": "all_scalar",
            "result": encode(module.broadcast_logic(km.q(5), False, km.xb(2))),
        }
    )
    broadcasts.append({"args": "empty", "result": encode(module.broadcast_logic())})
    observations = []
    sim = km.Simulator(batch_size=64, debug_print=False)
    for index in (
        km.q(0),
        km.b(1),
        km.xb(2),
        km.xbool(False),
        km.xbool(True),
        False,
        True,
    ):
        for pattern in constants.values():
            written = module.store64(sim, index, pattern)
            observations.append(
                {
                    "index": encode(index),
                    "pattern": pattern,
                    "written": written,
                    "read": module.read64(sim, index),
                }
            )
    output = Path(__file__).resolve().parents[1] / "data/physical_operand_cases.json"
    output.write_text(
        json.dumps(
            {
                "constants": constants,
                "cases": cases,
                "read_write": observations,
                "broadcasts": broadcasts,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
