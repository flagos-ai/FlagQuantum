# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Execute complete locked fuzz assertion paths and retain exact diagnostics."""

import hashlib
import importlib.util
import json
import random
from pathlib import Path
from typing import Any, cast

import kickmix as km


def main() -> None:
    assert km.__version__ == "0.1" and hasattr(km, "Simulator")
    assert hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest() == (
        "5e849075b49d895cca761fae864c8561b54362a6158fbfcabe7e9eb6e96d789e"
    )
    source = Path(km.__file__).parent / "src/kickmix/py/circuit/subs/fuzz_test_util.py"
    raw = source.read_bytes()
    assert (
        hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        == "7b2d1049f67198865c8af0a039dfc8eee1fcd253"
    )
    spec = importlib.util.spec_from_file_location("frozen_register_fuzz", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    named = 'REGISTER r0 "target"\nAPPEND_TO_REGISTER q0 r0\n'
    cases: list[dict[str, Any]] = []
    for shots in (0, 1, 3, 256, 257, 513):
        cases.append(
            {
                "circuit": named + "X q0",
                "code": "target ^= 1",
                "shots": shots,
                "samples": [{"target": 0}, {"target": 1}],
            }
        )
    for code in ("Z(target)\ntarget ^= 1", "\n\r  Z(target)\n  target ^= 1"):
        cases.append(
            {
                "circuit": named + "Z q0\nX q0",
                "code": code,
                "shots": 257,
                "samples": [{"target": 0}, {"target": 1}],
            }
        )
    cases.extend(
        [
            {"circuit": "NEG", "code": "neg()", "shots": 3},
            {"circuit": named + "CX q0 q1\nCX q0 q1", "code": "", "shots": 257},
            {
                "circuit": named + "X q1",
                "code": "",
                "shots": 1,
                "samples": [{"target": 0}],
            },
            {
                "circuit": named + "X q0",
                "code": "target ^= delta",
                "shots": 3,
                "context": {"delta": 0},
                "samples": [{"target": 0}],
            },
            {
                "circuit": named + "Z q0",
                "code": "",
                "shots": 1,
                "samples": [{"target": 1}],
            },
            {
                "circuit": named + "Z q0",
                "code": "",
                "shots": 1,
                "samples": [{"target": 1}],
                "ignore_phase": True,
            },
            {
                "circuit": named + "X q0\nZ q0\nX q1",
                "code": "",
                "shots": 1,
                "samples": [{"target": 0}],
                "context": {"note": "all faults"},
            },
            {"circuit": named, "code": "target = 0", "shots": 5},
            {"circuit": named, "code": "", "shots": 1, "samples": [{"wrong": 0}]},
            {
                "circuit": named,
                "code": "del target",
                "shots": 1,
                "samples": [{"target": 0}],
            },
            {
                "circuit": 'REGISTER r0 "target"\nAPPEND_TO_REGISTER q0 r0\nAPPEND_TO_REGISTER b0 r0\nX q0\nBIT_INVERT b0',
                "code": "target ^= 3",
                "shots": 257,
                "samples": [{"target": k} for k in range(4)],
            },
            {
                "circuit": 'REGISTER r0 "left"\nAPPEND_TO_REGISTER q0 r0\nREGISTER r1 "right"\nAPPEND_TO_REGISTER q0 r1',
                "code": "left = right",
                "shots": 3,
                "samples": [{"right": 1, "left": 0}],
            },
            {
                "circuit": 'REGISTER r0 "target"\nAPPEND_TO_REGISTER q0 r0\nREGISTER r1 "control"\nAPPEND_TO_REGISTER b0 r1\nX q0 if b0',
                "code": "if control:\n    target ^= 1",
                "shots": 257,
                "samples": [{"target": 0, "control": 0}, {"target": 1, "control": 1}],
            },
            {"circuit": named + "HMR q1 b1", "code": "", "shots": 513},
            {"circuit": "", "code": "", "shots": 1},
        ]
    )
    for case in cases:
        position = 0
        samples = cast(list[dict[str, int]] | None, case.get("samples"))

        def sampler(values: list[dict[str, int]] | None = samples) -> dict[str, int]:
            nonlocal position
            assert values is not None
            sample = values[position % len(values)]
            position += 1
            return sample

        random.seed(71)
        try:
            module.assert_fuzz_testing_acts_like(
                km.Circuit(case["circuit"]),
                case["code"],
                shots=case["shots"],
                context=case.get("context"),
                input_sampler=sampler if samples is not None else None,
                ignore_phase=case.get("ignore_phase", False),
            )
            case["result"] = "passed"
        except (AssertionError, KeyError, NameError) as error:
            case["exception"] = type(error).__name__
            case["result"] = str(error)
    output = Path(__file__).resolve().parents[1] / "data/physical_register_fuzz.json"
    output.write_text(json.dumps(cases, indent=2) + "\n")


if __name__ == "__main__":
    main()
