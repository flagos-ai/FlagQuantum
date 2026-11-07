"""Run python -m examples.algorithms.classical_action_dispatch.

Expected: recursive numerical dispatch agrees with append-controlled X then Z;
8 T, 2 measurements, 1 reused work wire and 5 declared wires.
"""

from collections import Counter
from itertools import product

import numpy as np
import torch

from examples.algorithms.classical_controlled_values import (
    WordControlledPaulis,
    x_child,
    z_child,
    z_phase,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.classical_action import evaluate_classical_action
from flagquantum.simulation.classical_control_patterns import normalize_control_patterns
from flagquantum.simulation.classical_controlled_values import (
    controlled_basis_phase,
    controlled_classical_values,
)
from flagquantum.simulation.classical_measurement import apply_classical_phase


def _no_phase(**values: object) -> complex | None:
    return None


def _unavailable_decomposition(
    values: dict[str, object],
) -> tuple[dict[str, object], complex | None]:
    raise NotImplementedError("These numerical leaves have no fallback in this example")


def evaluate_word_paulis(
    values: dict[str, object], *, inverse: bool = False
) -> tuple[dict[str, object], complex]:
    """Evaluate two existing controlled callbacks through the shared dispatcher."""
    patterns = normalize_control_patterns(2)

    def leaf(
        inputs: dict[str, object], kind: str
    ) -> tuple[dict[str, object], complex | None]:
        child = x_child if kind == "x" else z_child
        child_phase = _no_phase if kind == "x" else z_phase

        def value_action(**bound: object) -> object:
            return controlled_classical_values(
                bound, ("ctrl",), ("q",), patterns, child
            )

        def phase_action(**bound: object) -> complex | None:
            return controlled_basis_phase(
                bound, ("ctrl",), ("q",), patterns, child_phase
            )

        return evaluate_classical_action(
            inputs,
            value_action,
            phase_action,
            _unavailable_decomposition,
            operation_name=f"Controlled{kind.upper()}",
        )

    def compose(inputs: dict[str, object]) -> tuple[dict[str, object], complex | None]:
        phase: complex = 1.0
        for kind in (("z", "x") if inverse else ("x", "z")):
            inputs, contribution = leaf(inputs, kind)
            phase = apply_classical_phase(phase, contribution, {})
        return inputs, phase

    outputs, phase = evaluate_classical_action(
        values,
        lambda **bound: NotImplemented,
        _no_phase,
        compose,
        operation_name="WordControlledPaulis",
    )
    assert phase is not None
    return outputs, phase


def main() -> None:
    grid = torch.arange(1, 17, dtype=torch.float64)
    amplitudes = torch.complex(grid, 3 * grid - 5)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(32, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for word, target, spectator in product(range(4), (0, 1), (0, 1)):
        output, phase = evaluate_word_paulis({"ctrl": word, "q": target})
        final_target = output["q"]
        assert isinstance(final_target, (int, np.integer))
        before = (word << 3) | (target << 2) | spectator
        after = (word << 3) | (int(final_target) << 2) | spectator
        initial[before] = amplitudes[(word << 2) | (target << 1) | spectator]
        expected[after] = phase * initial[before]
    circuit = DynamicCircuit(5, inputs=initial, dtype=torch.complex128)
    circuit.append(WordControlledPaulis())
    result = run_dynamic(circuit, shots=8, seed=107, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states, expected.expand(8, -1), rtol=0, atol=2e-12
    )
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    assert counts["t"] + counts["tdg"] == 8 and counts["measure"] == 2
    circuit.append(WordControlledPaulis(inverse=True))
    restored = run_dynamic(circuit, shots=8, seed=109, strategy="trajectory")
    torch.testing.assert_close(
        restored.final_states, initial.expand(8, -1), rtol=0, atol=2e-12
    )
    print("Direct and recursive callbacks preserve full complex amplitudes and inverse")
    print("Emitted: 8 T, 2 measurements, 1 reused work, 5 declared wires")
    print("Model: 8 T equivalents, 2 measurements, 4 peak logical wires")


if __name__ == "__main__":
    main()
