# SPDX-License-Identifier: Apache-2.0
# Source observations from locked Kickmix Python binding scaffolding.

"""Observe all six extension storage families through their actual consumers."""

import hashlib
import json
from pathlib import Path

import kickmix as km


def main() -> None:
    assert km.__version__ == "0.1" and hasattr(km, "Simulator")
    assert hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest() == (
        "5e849075b49d895cca761fae864c8561b54362a6158fbfcabe7e9eb6e96d789e"
    )
    cases = []
    for count in (1, 3, 64, 257):
        simulator = km.Simulator(batch_size=count, debug_print=False)
        operands = [km.q(5), km.b(3), km.xb(2), km.xbool(False), km.xbool(True)]
        mask = sum(1 << lane for lane in range(count) if lane % 3 == 1)
        for operand in operands[:2]:
            simulator.write_across_shots(operand, mask)
        array = km.array(operands)
        circuit = km.Circuit('APPEND_TO_REGISTER q5 r0\nREGISTER r0 "data"')
        simulator.use_same_registers_as(circuit)
        cases.append(
            {
                "batch_size": count,
                "input_mask": mask,
                "scalar": [
                    simulator.read_across_shots(v, out=int) for v in operands[:2]
                ],
                "array": [str(v) for v in array],
                "reverse": [str(v) for v in array[::-1]],
                "register": simulator.read_across_shots(km.r(0), out=int),
                "name": simulator.read_across_shots("data", out=int),
            }
        )
    destination = (
        Path(__file__).resolve().parents[1] / "data/python_binding_scaffolding.json"
    )
    destination.write_text(json.dumps(cases, indent=2) + "\n")


if __name__ == "__main__":
    main()
