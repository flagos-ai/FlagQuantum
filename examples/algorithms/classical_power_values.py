"""Run python -m examples.algorithms.classical_power_values.

Expected for two repetitions: 16 T, 4 measurements, 1 reused work wire,
5 declared wires; original model peak 4 logical wires, excluding the spectator.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch

import flagquantum as fq
from examples.algorithms.classical_action_dispatch import evaluate_word_paulis
from examples.algorithms.classical_controlled_values import WordControlledPaulis
from flagquantum.compiler.registers.power import register_power_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.classical_power import power_classical_values
from flagquantum.simulation.classical_registers import (
    bind_integer_register_values,
    collect_integer_register_value,
)


@dataclass(frozen=True)
class PoweredWordPaulis:
    repetitions: int = 2
    inverse: bool = False

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        body = DynamicCircuit(circuit.n_qubits)
        body.append(WordControlledPaulis(inverse=self.inverse))
        gates = register_power_ir(body.to_ir(), self.repetitions)
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def evaluate_powered_word(
    values: dict[str, object], repetitions: int, *, inverse: bool = False
) -> tuple[dict[str, object], complex]:
    """Bind each complete child result before accumulating its resolved phase."""
    assigned: dict[str, object] = {}

    def assign(outputs: Mapping[str, object]) -> None:
        for name, width in (("ctrl", 2), ("q", 1)):
            bound = bind_integer_register_values(name, outputs, width)
            assigned[name] = collect_integer_register_value(bound, width)

    return power_classical_values(
        values,
        repetitions,
        lambda inputs: evaluate_word_paulis(inputs, inverse=inverse),
        assign,
    )


def main() -> None:
    grid = torch.arange(1, 17, dtype=torch.float64)
    amplitudes = torch.complex(grid, 3 * grid - 5)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(32, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for word, target, spectator in product(range(4), (0, 1), (0, 1)):
        output, phase = evaluate_powered_word({"ctrl": word, "q": target}, 2)
        final_target = output["q"]
        assert isinstance(final_target, (int, np.integer))
        before = (word << 3) | (target << 2) | spectator
        after = (word << 3) | (int(final_target) << 2) | spectator
        initial[before] = amplitudes[(word << 2) | (target << 1) | spectator]
        expected[after] = phase * initial[before]
    circuit = DynamicCircuit(5, inputs=initial, dtype=torch.complex128)
    circuit.append(PoweredWordPaulis())
    result = run_dynamic(circuit, shots=8, seed=157, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states, expected.expand(8, -1), rtol=0, atol=2e-12
    )
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    assert counts["t"] + counts["tdg"] == 16 and counts["measure"] == 4
    circuit.append(PoweredWordPaulis(inverse=True))
    restored = run_dynamic(circuit, shots=8, seed=163, strategy="trajectory")
    torch.testing.assert_close(
        restored.final_states, initial.expand(8, -1), rtol=0, atol=2e-12
    )
    print(
        "Repeated numerical values/phases and inverse preserve full complex amplitudes"
    )
    print("Emitted: 16 T, 4 measurements, 1 reused work, 5 declared wires")
    print("Source model: 16 T equivalents, 4 measurements, 4 peak logical wires")


if __name__ == "__main__":
    main()
