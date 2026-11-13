"""Independent Circuit and dense matrices, including absolute RZ and SX phases."""

import math

import numpy as np
import torch
from numpy.typing import NDArray

import flagquantum as fq

from .ansatz import Sequence


def gate_matrix(name: str) -> NDArray[np.complex128]:
    if name == "id":
        return np.eye(2, dtype=np.complex128)
    if name == "rz":
        return np.diag(np.array([-1j, 1j], dtype=np.complex128))
    if name == "sx":
        return np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex128) / 2
    if name == "x":
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)
    raise ValueError("unknown Clifford gate")


def circuit_reference(
    gates: Sequence, *, with_inverse: bool = True, initial: int = 0
) -> torch.Tensor:
    circuit = fq.Circuit(1, dtype=torch.complex128)
    if initial == 1:
        circuit.x(0)
    elif initial == 2:
        circuit.h(0).rz(0, theta=math.pi / 2)
    elif initial != 0:
        raise ValueError("initial state must be zero, one or coherent")
    for gate in gates:
        if gate == "id":
            circuit.any(0, matrix=torch.eye(2, dtype=torch.complex128))
        elif gate == "rz":
            circuit.rz(0, theta=math.pi)
        elif gate == "sx":
            circuit.any(
                0,
                matrix=torch.tensor(
                    [[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=torch.complex128
                )
                / 2,
            )
        else:
            circuit.x(0)
    if with_inverse:
        for gate in reversed(gates):
            matrix = torch.from_numpy(gate_matrix(gate).conj().T.copy())
            circuit.any(0, matrix=matrix)
    state = circuit.state()
    assert isinstance(state, torch.Tensor)
    return state.flatten()


def dense_reference(
    gates: Sequence, *, with_inverse: bool = True, initial: int = 0
) -> NDArray[np.complex128]:
    if initial == 0:
        state = np.array([1, 0], dtype=np.complex128)
    elif initial == 1:
        state = np.array([0, 1], dtype=np.complex128)
    elif initial == 2:
        state = np.array(
            [np.exp(-1j * math.pi / 4), np.exp(1j * math.pi / 4)], dtype=np.complex128
        ) / np.sqrt(2)
    else:
        raise ValueError("initial state must be zero, one or coherent")
    for gate in gates:
        state = gate_matrix(gate) @ state
    if with_inverse:
        for gate in reversed(gates):
            state = gate_matrix(gate).conj().T @ state
    return state
