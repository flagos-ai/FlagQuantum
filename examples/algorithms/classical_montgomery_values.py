"""Run python -m examples.algorithms.classical_montgomery_values.

Inspect numerical Montgomery history and bind the matching 21-wire append path.
"""

from collections import Counter

from examples.algorithms.register_windowed_mod_mul import (
    WindowedMontgomeryMultiplication,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.simulation.classical_modular_multiplication import (
    controlled_modular_multiply_constant_classical_values,
    modular_double_classical_value,
)
from flagquantum.simulation.classical_montgomery_multiplication import (
    montgomery_multiply_classical_values,
    montgomery_window_classical_values,
)


def main() -> None:
    assert modular_double_classical_value(2, 3) == 1
    assert controlled_modular_multiply_constant_classical_values(1, 2, 2, 3) == (1, 1)
    values = montgomery_multiply_classical_values(1, 2, 3, 1, 3)
    assert int(values["target"]) == 1 and int(values["qrom_indices"]) == 2
    assert int(values["reduced"]) == 0
    target, history = montgomery_window_classical_values(1, 2, 0, 0, 3, 1, 3)
    for offset in range(1, 3):
        target, history = montgomery_window_classical_values(
            1 >> offset, 2, target, history, 3, 1, 3
        )
    assert target == values["target"] and history == values["qrom_indices"]
    assert montgomery_multiply_classical_values(
        1,
        2,
        3,
        1,
        3,
        target=values["target"],
        history=values["qrom_indices"],
        reduced=values["reduced"],
        uncompute=True,
    ) == {"x": 1, "y": 2}
    print(
        "Numerical target/history/reduction:",
        int(target),
        int(history),
        int(values["reduced"]),
    )

    circuit = DynamicCircuit(21)
    operation = WindowedMontgomeryMultiplication()
    circuit.append(operation)
    counts = Counter(gate.name for gate in circuit.to_ir().instructions)
    emitted = counts["t"] + counts["tdg"], counts["measure"], circuit.n_qubits
    assert emitted == (128, 32, 21)
    print("Emitted T/measurement/declared wires:", emitted)
    print(
        "Pinned n=3,w=1,p=3 model T-equivalent/measurement/logical wires:",
        (128, 32, 21),
    )
    print("Forward target ports:", operation.target_wires)
    circuit.append(WindowedMontgomeryMultiplication(adjoint=True))
    print(
        "Forward and exact adjoint IR constructed; this example does not execute the circuit"
    )


if __name__ == "__main__":
    main()
