# SPDX-License-Identifier: Apache-2.0
# Observations from the unchanged locked circuit_builder.pybind.cc and header.

"""Record builder lifecycle transactions using the original extension."""

import hashlib
import json
from pathlib import Path
from typing import Any

import kickmix as km

from tests.unit.fixtures.generate_physical_circuit_binding import observe


def main() -> None:
    assert km.__version__ == "0.1" and hasattr(km, "Simulator")
    assert hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest() == (
        "5e849075b49d895cca761fae864c8561b54362a6158fbfcabe7e9eb6e96d789e"
    )
    scenarios: list[list[list[Any]]] = [
        [
            ["max", 7],
            ["quantum", 2, "data"],
            ["classical", 2, "out"],
            ["alloc_q", 3],
            ["alloc_b", 3],
            ["free", [["q", 2], ["b", 2], ["q", 4], ["b", 4]]],
            ["alloc_q", 3],
            ["alloc_b", 3],
            ["max", 6],
            ["max", 5],
            ["alloc_q", 1],
            ["max", "inf"],
            ["free", [["q", 0]]],
            ["free", [["b", 0]]],
            ["free", [["q", 2], ["q", 2], ["q", 3]]],
            ["alloc_q", 1],
            ["free", [["b", 2], ["b", 2], ["b", 3]]],
            ["alloc_b", 1],
        ],
        [
            ["alloc_q", 4],
            ["alloc_b", 4],
            ["free", [["q", 0], ["q", 3], ["q", 1]]],
            ["free", [["b", 1], ["b", 0], ["b", 3]]],
            ["quantum", 2, "data"],
            ["classical", 2, "out"],
            ["alloc_q", 5],
            ["alloc_b", 5],
            ["free", [["q", 3], ["bool", 1], ["q", 1]]],
            ["alloc_q", 2],
        ],
        [
            ["max", 0],
            ["alloc_q", 0],
            ["alloc_b", 0],
            ["quantum", 0, "empty"],
            ["classical", 0, "empty_classical"],
            ["alloc_q", 1],
            ["alloc_b", 2],
            ["max", 3],
            ["quantum", 3, "data"],
            ["max", 2],
            ["free", [["q", 1]]],
            ["alloc_q", 0],
            ["max", (1 << 64) - 1],
            ["max", 1 << 63],
            ["max", (1 << 64) - 2],
            ["max", 1e100],
            ["max", float.fromhex("0x1.ffffffp+127")],
        ],
    ]
    for order in ([0, 1, 2], [2, 1, 0], [1, 0, 2]):
        scenarios.append(
            [
                ["alloc_q", 3],
                ["alloc_b", 3],
                ["free", [["q", wire] for wire in order]],
                ["free", [["b", wire] for wire in order]],
                ["alloc_q", 4],
                ["alloc_b", 4],
            ]
        )
    traces: list[dict[str, Any]] = []
    for actions in scenarios:
        builder = km.CircuitBuilder()
        observations = []
        for action in actions:
            result = None
            error = None
            try:
                match action:
                    case ["max", value]:
                        builder.set_max_qubits(
                            float("inf") if value == "inf" else value
                        )
                    case ["quantum", count, name]:
                        result = list(builder.create_quantum_register(count, name))
                    case ["classical", count, name]:
                        result = list(builder.create_classical_register(count, name))
                    case ["alloc_q", count]:
                        result = list(builder.alloc_qubits(count))
                    case ["alloc_b", count]:
                        result = list(builder.alloc_bits(count))
                    case ["free", operands]:
                        values = [
                            (
                                km.q(value)
                                if kind == "q"
                                else km.b(value) if kind == "b" else bool(value)
                            )
                            for kind, value in operands
                        ]
                        builder.free(values)
                    case _:
                        raise AssertionError(action)
            except ValueError as exception:
                error = str(exception)
            observations.append(
                {
                    "result": None if result is None else [str(x) for x in result],
                    "error": error,
                    "max": str(builder.max_qubits),
                    "free": str(builder.num_free_qubits),
                    "allocated": builder.num_allocated_qubits,
                    "circuit": observe(builder.finish_circuit()),
                }
            )
        traces.append({"actions": actions, "observations": observations})

    builder = km.CircuitBuilder()
    q = builder.create_quantum_register(4, "data")
    reused = builder.mark("reused")
    for _ in range(2):
        with reused:
            builder.ccx(q[0], q[1], q[2])
    with builder.mark("outer"), builder.mark("inner <0 & 1>"):
        builder.ccx(q[1], q[2], q[3])
    with builder.mark(""):
        builder.z_pow(q[0], "0.1")
    try:
        with builder.mark('failed "block"\n'):
            builder.x(q[3])
            raise RuntimeError("body failure")
    except RuntimeError:
        pass
    first, second = builder.mark("first"), builder.mark("second")
    first.__enter__()
    second.__enter__()
    first.__exit__(None, None, None)
    second.__exit__(None, None, None)
    builder.debug_print([q[0], q[3]])
    value = {
        "revision": "9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453",
        "traces": traces,
        "marks": observe(builder.finish_circuit()),
    }
    target = Path(__file__).parents[1] / "data" / "physical_builder_binding.json"
    target.write_text(json.dumps(value, indent=2) + "\n")
    print(f"Recorded {sum(len(trace['actions']) for trace in traces)} transactions")


if __name__ == "__main__":
    main()
