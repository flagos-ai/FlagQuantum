"""Run python -m examples.algorithms.numerical_value_dimensions."""

from collections import Counter
from dataclasses import dataclass
from itertools import product
from math import prod

import numpy as np
import torch

import flagquantum as fq
from flagquantum.compiler.registers.control_predicate_resource_model import (
    control_predicate_resource_formula,
)
from flagquantum.compiler.registers.pauli_gate import pauli_gate_ir
from flagquantum.compiler.resource_reporting import resource_model_report
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.classical_control_activation import (
    classical_controls_active,
)
from flagquantum.simulation.classical_control_patterns import normalize_control_patterns
from flagquantum.simulation.value_dimensions import (
    SymbolicArrayShape,
    SymbolicSequenceLength,
    value_length,
    value_shape,
)


@dataclass(frozen=True)
class ShapedControlledX:
    """Use known row-major 2x2 activation values with the existing predicate emitter."""

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        patterns = normalize_control_patterns(np.array([[0, 1], [1, 0]]))
        values = tuple(int(bit) for bit in np.asarray(patterns[0]).flat)
        gates = pauli_gate_ir(
            "x",
            4,
            controls=(0, 1, 2, 3),
            values=values,
            workspace=(5, 6, 7),
            classical_bits=(0, 1, 2),
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    unknown = SymbolicArrayShape((2, 2))
    assert value_shape(unknown) == (2, 2) and value_length(unknown) == 2
    assert value_length(SymbolicSequenceLength(4)) == 4
    try:
        classical_controls_active((np.zeros((2, 2), dtype=int),), (unknown,))
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown values cannot be executed classically")
    # This model covers predicate computation, not its erase or controlled body.
    model = resource_model_report(
        control_predicate_resource_formula(prod(unknown.shape))
    )
    assert model["t_equivalents"] == 12 and model["peak_logical_qubits"] == 7
    print("Shape-only predicate compute model:", model["logical_gates"])
    patterns = normalize_control_patterns(np.array([[0, 1], [1, 0]]))
    grid = torch.arange(1, 65, dtype=torch.float64)
    weights = torch.complex(grid, 3 * grid - 5)
    weights /= weights.norm()
    initial = torch.zeros(512, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for word, target, spectator in product(range(16), (0, 1), (0, 1)):
        controls = np.array([(word >> i) & 1 for i in (3, 2, 1, 0)]).reshape(2, 2)
        active = classical_controls_active((controls,), patterns)
        before = (word << 5) | (target << 4) | spectator
        after = (word << 5) | ((target ^ active) << 4) | spectator
        initial[before] = weights[(word << 2) | (target << 1) | spectator]
        expected[after] = initial[before]
    circuit = DynamicCircuit(9, inputs=initial, dtype=torch.complex128)
    op = ShapedControlledX()
    circuit.append(op)
    result = run_dynamic(circuit, shots=8, seed=89, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states, expected.expand(8, -1), rtol=0, atol=2e-12
    )
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    assert counts["t"] + counts["tdg"] == 12 and counts["measure"] == 3
    circuit.append(op)
    restored = run_dynamic(circuit, shots=8, seed=97, strategy="trajectory")
    torch.testing.assert_close(
        restored.final_states, initial.expand(8, -1), rtol=0, atol=2e-12
    )
    print("Known-value controlled X: 12 emitted T, 3 measurements, 3 work, 9 wires")
    print("Full complex amplitudes/inverse agree; work zero; spectator preserved")
    print("Unknown-value model and known-value execution are separate evidence bases")


if __name__ == "__main__":
    main()
