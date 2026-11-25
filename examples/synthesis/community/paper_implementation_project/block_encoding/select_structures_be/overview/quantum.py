"""Compose native PREPARE, structured SELECT and inverse PREPARE."""

import math

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, prepare_amplitudes, qfunc

from .model import DTYPE, Structure


@qfunc
def prepare(values: list[float], q: Output[QArray[QBit]]) -> None:
    prepare_amplitudes(values, 0, q)


def _preparation(values: tuple[float, ...]) -> fq.Circuit:
    width = len(values).bit_length() - 1
    program = synth.synthesize(
        prepare, classical_inputs={"values": values}, max_qubits=width, max_workspace=0
    )
    return fq.Circuit.from_ir(program.circuit, dtype=DTYPE)


def _structured_select(structure: Structure) -> torch.Tensor:
    size = structure.matrix.shape[0]
    flag_states = 2 << int(structure.boundary_flag)
    branch_size = size * flag_states
    dimension = (1 << structure.selector_bits) * branch_size
    matrix = torch.eye(dimension, dtype=DTYPE)
    for key, term in structure.addresses:
        group = slice(key * branch_size, (key + 1) * branch_size)
        matrix[group, group] = 0
        base = key * branch_size
        for col, (row, coefficient, invalid) in enumerate(
            zip(term.rows, term.weights, term.invalid, strict=True)
        ):
            # Load the signed coefficient onto the zero state of the value flag.
            cross = math.sqrt(max(0.0, 1.0 - coefficient**2))
            for boundary in range(flag_states // 2):
                input_zero = base + (2 * boundary) * size + col
                output_zero = base + (2 * (boundary ^ invalid)) * size + row
                matrix[output_zero, input_zero] = matrix[
                    output_zero + size, input_zero + size
                ] = coefficient
                matrix[output_zero + size, input_zero] = cross
                matrix[output_zero, input_zero + size] = -cross
    return matrix


def build_encoding(structure: Structure, data: tuple[float, ...]) -> fq.Circuit:
    """Build the source-width block with uniform weights over populated addresses."""
    size = structure.matrix.shape[0]
    if len(data) != size:
        raise ValueError("Input must match the source matrix dimension")
    bits = size.bit_length() - 1
    selector = tuple(range(structure.selector_bits))
    flags = structure.selector_bits + 1 + int(structure.boundary_flag)
    system = tuple(range(flags, flags + bits))
    weights = [0.0] * (1 << structure.selector_bits)
    for key, _ in structure.addresses:
        weights[key] = 1 / math.sqrt(len(structure.addresses))
    amplitudes = _preparation(tuple(weights))
    circuit = fq.Circuit(flags + bits, dtype=DTYPE)
    circuit.compose(_preparation(data), qubits=system).compose(
        amplitudes, qubits=selector
    )
    circuit.any(
        *range(flags + bits),
        matrix=_structured_select(structure),
        name="structured_select"
    )
    circuit.compose(amplitudes.adjoint(), qubits=selector)
    return circuit
