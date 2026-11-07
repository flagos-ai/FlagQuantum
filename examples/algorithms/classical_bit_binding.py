"""Run python -m examples.algorithms.classical_bit_binding."""

from collections import Counter
from dataclasses import dataclass
from functools import partial
from itertools import product

import torch

import flagquantum as fq
from flagquantum.compiler.registers.register_bits import join_register, split_register
from flagquantum.compiler.registers.register_cast import RegisterCast
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.simulation.classical_bit_binding import (
    allocate_classical_value,
    cast_classical_value,
    free_classical_value,
    join_classical_value,
    split_classical_value,
)
from flagquantum.simulation.classical_pauli_values import x_classical_value
from flagquantum.simulation.integer_encoding import integer_from_bits, integer_to_bits


@dataclass(frozen=True)
class ReversedViewX:
    """Flip the leading bit of a reversed, signed three-bit register view."""

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        parts = split_register((0, 1, 2))
        packed = join_register(tuple(reversed(parts)))
        view = RegisterCast(3, 3).bind(packed)
        return circuit.compose(
            CircuitIR(circuit.n_qubits, (Instruction("x", (view[0],)),))
        )


def main() -> None:
    encode = partial(integer_to_bits, width=3)
    decode = partial(integer_from_bits, width=3)
    signed_encode = partial(integer_to_bits, width=3, signed=True)
    signed_decode = partial(integer_from_bits, width=3, signed=True)
    assert allocate_classical_value(3, decode) == 0
    assert free_classical_value(0) == {}
    grid = torch.arange(1, 17, dtype=torch.float64)
    initial = torch.complex(grid, 3 * grid - 5)
    initial /= initial.norm()
    expected = torch.zeros_like(initial)
    for data, spectator in product(range(8), (0, 1)):
        bits = split_classical_value(data, encode)[::-1].copy()
        bits[0] = x_classical_value(bits[0])
        unsigned = join_classical_value(bits, decode)
        signed = cast_classical_value(unsigned, encode, signed_decode)
        physical = join_classical_value(
            split_classical_value(signed, signed_encode)[::-1], decode
        )
        expected[(physical << 1) | spectator] = initial[(data << 1) | spectator]
        assert free_classical_value(data, dirty=True) == {}
    circuit = fq.Circuit(4, inputs=initial, dtype=torch.complex128)
    op = ReversedViewX()
    circuit.append(op)
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)
    assert Counter(g.name for g in circuit.to_ir().instructions) == {"x": 1}
    circuit.append(op)
    torch.testing.assert_close(circuit.state().flatten(), initial, rtol=0, atol=2e-12)
    print(
        "Raw codecs and register views agree on complete complex amplitudes and inverse"
    )
    print("Bookkeeping emitted: 0 gates, 0 T, 0 measurements; explicit operation: 1 X")
    print("Source bookkeeping models: 0 gate cost; 3 logical data wires plus spectator")
    print(
        "Numerical Free consumes a value; it does not reset or release physical memory"
    )


if __name__ == "__main__":
    main()
