"""Run python -m examples.algorithms.classical_auto_partition_values."""

from collections import Counter
from dataclasses import dataclass
from functools import partial
from itertools import product

import torch

import flagquantum as fq
from flagquantum.compiler.registers.arithmetic.addition import add_adjoint_ir, add_ir
from flagquantum.compiler.registers.layout.auto_partition import AutoPartition
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.classical_addition import register_add_classical_values
from flagquantum.simulation.classical_auto_partition import (
    PackedValueBinding,
    auto_partition_classical_values,
)
from flagquantum.simulation.classical_partition import PartitionValuePort
from flagquantum.simulation.integer_encoding import (
    integer_array_from_bits,
    integer_array_to_bits,
    integer_from_bits,
    integer_to_bits,
)


def value_binding() -> PackedValueBinding:
    ports = tuple(
        PartitionValuePort(
            name,
            width,
            (),
            partial(integer_from_bits, width=width),
            partial(integer_array_from_bits, width=width),
            partial(integer_array_to_bits, width=width),
        )
        for name, width in (("a", 2), ("_unused0_1", 1), ("b", 2))
    )
    return PackedValueBinding(
        "packed",
        ports,
        partial(integer_to_bits, width=5),
        partial(integer_from_bits, width=5),
    )


def add_child(values: dict[str, object]) -> dict[str, object]:
    left, right = values["a"], values["b"]
    assert isinstance(left, int) and isinstance(right, int)
    left, right = register_add_classical_values(left, right, 2)
    return {"a": left, "b": right}


@dataclass(frozen=True)
class PackedAdd:
    """Add two packed two-bit words while retaining the interleaved unused bit."""

    inverse: bool = False

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        layout = AutoPartition((("packed", (("a", 2), (None, 1), ("b", 2))),))
        inputs, unused = layout.bind({"packed": (0, 1, 2, 3, 4)})
        emit = add_adjoint_ir if self.inverse else add_ir
        gates = emit(inputs["a"], inputs["b"], (5,), (0,))
        assert layout.restore(inputs, unused) == {"packed": (0, 1, 2, 3, 4)}
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    binding = value_binding()
    grid = torch.arange(1, 65, dtype=torch.float64)
    amplitudes = torch.complex(grid, 3 * grid - 5)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(128, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for word, spectator in product(range(32), (0, 1)):
        output = auto_partition_classical_values(
            {"packed": word}, (binding,), ("a", "b"), add_child
        )["packed"]
        assert isinstance(output, int)
        initial[(word << 2) | spectator] = amplitudes[2 * word + spectator]
        expected[(output << 2) | spectator] = amplitudes[2 * word + spectator]
    circuit = DynamicCircuit(7, inputs=initial, dtype=torch.complex128)
    circuit.append(PackedAdd())
    result = run_dynamic(circuit, shots=8, seed=61, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states, expected.expand(8, -1), rtol=0, atol=2e-12
    )
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    assert counts["t"] + counts["tdg"] == 4 and counts["measure"] == 1
    circuit.append(PackedAdd(inverse=True))
    restored = run_dynamic(circuit, shots=8, seed=67, strategy="trajectory")
    torch.testing.assert_close(
        restored.final_states, initial.expand(8, -1), rtol=0, atol=2e-12
    )
    print(
        "Packed value composition and measured gates agree on complete complex amplitudes"
    )
    print("Unused bit and spectator preserved; inverse restores inputs; work zero")
    print("Emitted: 4 T, 1 measurement, 1 work, 7 declared wires")
    print("Source model: 4 T equivalents, 1 measurement, 6 peak logical wires")


if __name__ == "__main__":
    main()
