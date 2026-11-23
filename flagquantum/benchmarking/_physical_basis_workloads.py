# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix sim/sim.perf.cc.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Compose complete physical sampling benchmark bodies over numerical owners."""

from collections.abc import Callable
from typing import Literal

from ..compiler.formats.kmx_circuit import KmxCircuit
from ..core._binary_matrix import BinaryMatrix
from ..simulation._basis_simulation import PackedBasisSimulation
from ..simulation._packed_word import pack_lanes
from ..simulation.physical_basis_rng import _mt19937_64

SimulationWorkload = Literal["sample", "counting", "populates"]


def simulation_workload(
    circuit: KmxCircuit, width: int, mode: SimulationWorkload
) -> tuple[PackedBasisSimulation, Callable[[], None]]:
    """Return owned persistent state and one original clear/apply workload."""
    if mode not in ("sample", "counting", "populates"):
        raise ValueError("Unknown physical simulation workload")
    sim = PackedBasisSimulation(width, _mt19937_64(0), counting=mode == "counting")
    sim.configure_for(circuit)

    def body() -> None:
        sim.clear_for_shot()
        if mode == "populates":
            for wire in range(sim.qubits):
                sim.state.qubits[wire] = pack_lanes(
                    (next(sim.rng) for _ in range(width // 64)), 64
                )
            sim.copy_state_into_register_buffer()
            sim.register_buffer, sim.expected_buffer = (
                sim.expected_buffer,
                sim.register_buffer,
            )
        sim.apply(circuit)
        if mode == "populates":
            sim.copy_state_into_register_buffer()

    return sim, body


def transpose_workload() -> Callable[[], None]:
    """Compose the complete 64 by 64 workload with the existing Core transpose."""
    data = BinaryMatrix((1, *([0] * 63)), 64)

    def body() -> None:
        nonlocal data
        data = data.transposed()

    return body
