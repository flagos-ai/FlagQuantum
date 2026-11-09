"""Measure a prepared three-bit Fourier word through circuit.append(operation).

Run python -m examples.algorithms.fourier_measurement [--inverse-qft] [--shots 64].
"""

import argparse
from collections import Counter
from math import pi

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.phase.qft_measurement import (
    kickmix_qft_measurement_program,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


def make_circuit(*, inverse_qft: bool = False) -> DynamicCircuit:
    value, dimension = 3, 8
    x = torch.arange(dimension, dtype=torch.float64)
    sign = -1 if inverse_qft else 1
    amplitudes = torch.exp(-sign * 2j * pi * value * x / dimension) / dimension**0.5
    initial = torch.zeros(16, dtype=torch.complex128)
    initial[::2] = amplitudes  # The last qubit is an untouched spectator.
    program = kickmix_qft_measurement_program(
        (0, 1, 2), (0, 1, 2), n_qubits=4, inverse_qft=inverse_qft
    )
    operation = FieldOperation(4, bind_reversible_phase_execution(program))
    circuit = DynamicCircuit(4, inputs=initial, dtype=torch.complex128)
    circuit.append(operation)
    return circuit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inverse-qft", action="store_true")
    parser.add_argument("--shots", type=int, default=64)
    args = parser.parse_args()
    if args.shots < 1:
        parser.error("shots must be positive")
    result = run_dynamic(
        make_circuit(inverse_qft=args.inverse_qft),
        shots=args.shots,
        seed=41,
        strategy="batched",
    )
    words = result.classical_bits[:, :3].long() @ torch.tensor([4, 2, 1])
    counts = dict(sorted(Counter(words.tolist()).items()))
    states = result.final_states.reshape(args.shots, 8, 2)
    cleared = bool(torch.all(states[:, 1:].abs() < 1e-12))
    assert counts == {3: args.shots} and cleared
    print("Fourier words", counts)
    print("Measured target cleared", cleared)


if __name__ == "__main__":
    main()
