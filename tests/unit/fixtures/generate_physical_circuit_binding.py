# SPDX-License-Identifier: Apache-2.0
# Observations from locked Kickmix circuit.pybind.cc and circuit_test.py.

"""Record complete circuit-value binding operations from the locked extension."""

import hashlib
import json
from pathlib import Path

import kickmix as km


def observe(circuit: km.Circuit) -> dict[str, object]:
    simulator = km.Simulator(
        batch_size=3, seed=0, debug_print=False, counting_operations=True
    )
    for wire in range(circuit.num_qubits):
        simulator.write_across_shots(km.q(wire), (wire * 5 + 3) & 7)
    simulator.do(circuit)
    return {
        "text": str(circuit),
        "records": len(circuit),
        "qubits": circuit.num_qubits,
        "bits": circuit.num_bits,
        "registers": circuit.num_registers,
        "register_data": [
            [
                ["name", key] if isinstance(key, str) else ["index", key.id],
                [str(value) for value in values],
            ]
            for key, values in circuit.register_data.items()
        ],
        "reaction_depth": circuit.reaction_depth(),
        "text_diagram": circuit.text_diagram(),
        "static_counts": dict(circuit.operation_statistics()),
        "executed_counts": dict(simulator.operation_statistics()),
        "quantum": [
            simulator.read_across_shots(km.q(wire), out=int)
            for wire in range(circuit.num_qubits)
        ],
        "classical": [
            simulator.read_across_shots(km.b(bit), out=int)
            for bit in range(circuit.num_bits)
        ],
        "phases": [str(simulator.read_shot_phase(shot)) for shot in range(3)],
    }


def main() -> None:
    assert km.__version__ == "0.1" and hasattr(km, "Simulator")
    assert hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest() == (
        "5e849075b49d895cca761fae864c8561b54362a6158fbfcabe7e9eb6e96d789e"
    )
    circuits = [
        km.Circuit(),
        km.Circuit('REGISTER r0 "test"'),
        km.Circuit("CCX q0 q1 q2\nCZ q0 q1"),
        km.Circuit('REGISTER r0 "x"\nCCX q0 q1 q2\nCZ q0 q1'),
        km.Circuit('REGISTER r0 "y"\nX q2'),
        km.Circuit("X q2"),
        km.Circuit(
            'REGISTER r0 "test"\nAPPEND_TO_REGISTER q0 r0\n'
            'APPEND_TO_REGISTER q1 r0\nREGISTER r2 "empty"\n'
            "APPEND_TO_REGISTER q3 r3\nAPPEND_TO_REGISTER b0 r3\n"
            "HMR q0 b0\nZ q1 if b0\nZ_POW q2 0.25"
        ),
    ]
    observations = [observe(circuit) for circuit in circuits]
    additions = [
        [left, right, observe(a + b)]
        for left, a in enumerate(circuits)
        for right, b in enumerate(circuits)
    ]
    repetitions = [
        (index, count, observe(circuit * count))
        for index, circuit in enumerate(circuits)
        for count in (-(1 << 63), -999, -1, 0, 1, 2, 3, True, False)
    ]
    for index, count, expected in repetitions:
        assert observe(count * circuits[index]) == expected
    data = {
        "circuits": observations,
        "additions": additions,
        "repetitions": repetitions,
        "equalities": [[left == right for right in circuits] for left in circuits],
    }
    destination = (
        Path(__file__).resolve().parents[1] / "data/physical_circuit_binding.json"
    )
    destination.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
