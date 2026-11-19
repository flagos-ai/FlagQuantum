"""Define the original five-bit thermal QuSO model without running it by default."""

import math

import torch

import flagquantum as fq
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.verify_block_encoding.overview.quantum import (
    preparation,
)
from flagquantum.algorithms.primitives.qft import qft

from . import model, quantum

PHASE_BITS = 5
SOLVER_BITS = 9
WIDTH = len(model.EDGES) + PHASE_BITS + SOLVER_BITS


def conditional_solver(reciprocal: model.Reciprocal) -> fq.Circuit:
    """Prepare the heat vector and encode its inverse for each design."""
    base = quantum.conditional_thermal_encoding()
    inverse = base.adjoint()
    circuit = fq.Circuit(15, dtype=model.COMPLEX)
    # The first solver bit is retained in the scratch layout. The native
    # real-response projector combines the two signal roles on the next wire.
    signal_wire = 7
    block_wires = tuple(range(8, 13))
    data = (13, 14)
    targets = (*range(6), *block_wires, *data)
    heat = torch.tensor(model.HEAT, dtype=model.REAL)
    circuit.compose(preparation(heat / heat.norm()), qubits=data)
    circuit.h(signal_wire)
    phases = reciprocal.fit.phases - math.pi / 2
    phases[0] += math.pi / 4
    phases[-1] += math.pi / 4
    words = torch.arange(64)
    signs = 2 * (((words & 31) == 0) == (words >> 5 == 0)).to(model.REAL) - 1
    for i, angle in enumerate(phases):
        projector = torch.diag((1j * angle * signs).exp())
        circuit.any(signal_wire, *block_wires, matrix=projector)
        if i < phases.numel() - 1:
            circuit.compose(inverse if i % 2 == 0 else base, qubits=targets)
    return circuit.h(signal_wire)


def controlled_step(step: fq.Circuit) -> fq.Circuit:
    """Add a phase-estimation control to the known local simulator operations."""
    result = fq.Circuit(step.n_qubits + 1, dtype=model.COMPLEX)
    control = step.n_qubits
    controlled_matrices: dict[
        tuple[int, tuple[int, ...], tuple[int, ...], bool], torch.Tensor
    ] = {}
    for item in step.to_ir().instructions:
        if item.name == "any":
            matrix = item.matrix
            if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2:
                raise ValueError("Only explicit local matrices can be controlled")
            key = (
                matrix.data_ptr(),
                tuple(matrix.shape),
                matrix.stride(),
                matrix.is_conj(),
            )
            if key not in controlled_matrices:
                size = matrix.size(0)
                operator = torch.eye(2 * size, dtype=model.COMPLEX)
                operator[size:, size:] = matrix
                controlled_matrices[key] = operator
            result.any(control, *item.wires, matrix=controlled_matrices[key])
        else:
            width = len(item.wires)
            local = fq.Circuit(width, dtype=model.COMPLEX)
            local.gate(item.name, tuple(range(width)), params=dict(item.params))
            result.compose(local.control(1, width), qubits=(*item.wires, control))
    return result


def amplitude_estimation(reciprocal: model.Reciprocal) -> fq.Circuit:
    """Define the five-bit QAE stage used by the full QuSO model."""
    solver = conditional_solver(reciprocal)
    grover = fq.Circuit(solver.n_qubits, dtype=model.COMPLEX)
    good = torch.ones(128, dtype=model.COMPLEX)
    good[0] = -1
    grover.any(*range(len(model.EDGES), len(model.EDGES) + 7), matrix=torch.diag(good))
    grover.compose(solver.adjoint())
    zero = -torch.ones(512, dtype=model.COMPLEX)
    zero[0] = 1
    grover.any(*range(len(model.EDGES), solver.n_qubits), matrix=torch.diag(zero))
    grover.compose(solver)
    controlled = controlled_step(grover)

    design = tuple(range(len(model.EDGES)))
    phase = tuple(range(len(design), len(design) + PHASE_BITS))
    workspace = tuple(range(phase[-1] + 1, WIDTH))
    result = fq.Circuit(WIDTH, dtype=model.COMPLEX).compose(
        solver, qubits=(*design, *workspace)
    )
    for wire in phase:
        result.h(wire)
    for index, wire in enumerate(phase):
        for _ in range(1 << (PHASE_BITS - 1 - index)):
            result.compose(controlled, qubits=(*design, *workspace, wire))
    return result.compose(qft(PHASE_BITS, inverse=True), qubits=phase)


def build_full_quso(
    reciprocal: model.Reciprocal, gamma: float, beta: float
) -> fq.Circuit:
    """Return the full 20-bit design; the source leaves this run disabled."""
    if not all(map(math.isfinite, (gamma, beta))):
        raise ValueError("Full QuSO angles must be finite")
    oracle = amplitude_estimation(reciprocal)
    circuit = fq.Circuit(WIDTH, dtype=model.COMPLEX)
    for wire in range(len(model.EDGES)):
        circuit.h(wire)
    circuit.compose(oracle)
    phases = torch.arange(32, dtype=model.REAL) * math.pi / 32
    circuit.any(*range(6, 11), matrix=torch.diag((1j * gamma * phases.sin()).exp()))
    circuit.compose(oracle.adjoint())
    for wire in range(len(model.EDGES)):
        circuit.rx(wire, theta=beta)
    return circuit.compose(oracle)
