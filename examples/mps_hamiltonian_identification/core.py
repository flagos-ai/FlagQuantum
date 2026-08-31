"""Differentiable MPS model for non-uniform 1D Hamiltonian identification.

This development-evidence example intentionally imports experimental MPS
brickwork kernels. It is a research implementation, not a stable public-API
template; user-facing applications should start from ``fq.Module`` and
``fq.RuntimePolicy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

import flagquantum as fq
import flagquantum.backends as fqb
from flagquantum.simulation.mps_brickwork import (
    compiled_local_z_zz,
    run_batched_brickwork_mps,
)


@dataclass(frozen=True)
class Probe:
    """One product-state preparation observed after a discrete evolution time."""

    time_steps: int
    flipped_sites: tuple[int, ...]


def smooth_couplings(n_wires: int, *, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Return a deterministic, spatially non-uniform teacher Hamiltonian."""

    x = torch.linspace(0.0, 1.0, n_wires, device=device)
    bonds = torch.linspace(0.0, 1.0, n_wires - 1, device=device)
    coupling = 0.72 + 0.18 * torch.sin(2.0 * torch.pi * bonds) + 0.05 * torch.cos(5.0 * torch.pi * bonds)
    field = 0.43 + 0.13 * torch.cos(2.0 * torch.pi * x + 0.2) + 0.04 * torch.sin(6.0 * torch.pi * x)
    return coupling, field


def make_probes(n_wires: int, *, n_initial_states: int, time_steps: Iterable[int], seed: int) -> tuple[Probe, ...]:
    """Create reproducible sparse product-state/time probes."""

    if n_wires < 2 or n_initial_states < 1:
        raise ValueError("n_wires must be >= 2 and n_initial_states must be positive")
    times = tuple(int(value) for value in time_steps)
    if not times or any(value < 1 for value in times):
        raise ValueError("time_steps must contain positive integers")
    generator = torch.Generator().manual_seed(int(seed))
    probes = []
    width = max(1, min(4, n_wires // 16 or 1))
    for _ in range(n_initial_states):
        sites = torch.randperm(n_wires, generator=generator)[:width].sort().values.tolist()
        probes.extend(Probe(time_steps=value, flipped_sites=tuple(sites)) for value in times)
    return tuple(probes)


def build_trotter_circuit(
    coupling: torch.Tensor,
    field: torch.Tensor,
    probe: Probe,
    *,
    dt: float,
    device: torch.device | str,
) -> fq.Circuit:
    """Build a nearest-neighbour TFIM-style differentiable Trotter circuit.

    The learned effective Hamiltonian is H = sum J_i X_i X_{i+1} + sum h_i Y_i.
    A brickwork ordering keeps the circuit MPS-friendly while still propagating
    information across the chain over time.
    """

    n_wires = int(field.numel())
    if coupling.shape != (n_wires - 1,):
        raise ValueError("coupling must have shape (n_wires - 1,)")
    circuit = fq.Circuit(n_wires, device=device)
    for wire in probe.flipped_sites:
        circuit.x(int(wire))
    for _ in range(int(probe.time_steps)):
        for wire in range(n_wires):
            circuit.ry(wire, theta=2.0 * float(dt) * field[wire])
        for parity in (0, 1):
            for wire in range(parity, n_wires - 1, 2):
                circuit.rxx(wire, wire + 1, theta=2.0 * float(dt) * coupling[wire])
    return circuit


def build_batched_trotter_circuit(
    coupling: torch.Tensor,
    field: torch.Tensor,
    probes: tuple[Probe, ...],
    *,
    dt: float,
    device: torch.device | str,
) -> fq.Circuit:
    """Build one batched circuit for probes with an identical time depth."""

    if not probes or len({probe.time_steps for probe in probes}) != 1:
        raise ValueError("a probe batch must be non-empty and have one time depth")
    n_wires = int(field.numel())
    circuit = fq.Circuit(n_wires, bsz=len(probes), device=device)
    for wire in range(n_wires):
        angles = torch.tensor(
            [torch.pi if wire in probe.flipped_sites else 0.0 for probe in probes],
            dtype=field.dtype,
            device=device,
        )
        circuit.rx(wire, theta=angles)
    for _ in range(probes[0].time_steps):
        for wire in range(n_wires):
            circuit.ry(wire, theta=2.0 * float(dt) * field[wire])
        for parity in (0, 1):
            for wire in range(parity, n_wires - 1, 2):
                circuit.rxx(wire, wire + 1, theta=2.0 * float(dt) * coupling[wire])
    return circuit


def build_variable_time_batched_trotter_circuit(
    coupling: torch.Tensor,
    field: torch.Tensor,
    probes: tuple[Probe, ...],
    *,
    dt: float,
    device: torch.device | str,
) -> fq.Circuit:
    """Build one masked batch containing probes with different time depths."""

    if not probes:
        raise ValueError("probe batch must be non-empty")
    n_wires = int(field.numel())
    circuit = fq.Circuit(n_wires, bsz=len(probes), device=device)
    for wire in range(n_wires):
        angles = torch.tensor(
            [torch.pi if wire in probe.flipped_sites else 0.0 for probe in probes],
            dtype=field.dtype,
            device=device,
        )
        circuit.rx(wire, theta=angles)
    for layer in range(max(probe.time_steps for probe in probes)):
        active = torch.tensor(
            [1.0 if layer < probe.time_steps else 0.0 for probe in probes],
            dtype=field.dtype,
            device=device,
        )
        for wire in range(n_wires):
            circuit.ry(wire, theta=2.0 * float(dt) * field[wire] * active)
        for parity in (0, 1):
            for wire in range(parity, n_wires - 1, 2):
                circuit.rxx(
                    wire,
                    wire + 1,
                    theta=2.0 * float(dt) * coupling[wire] * active,
                )
    return circuit


def local_observables(state: fq.MPSState, observation_sites: Iterable[int]) -> torch.Tensor:
    """Return Z_i and nearest-neighbour ZZ_i measurements without dense state materialization."""

    sites = tuple(int(wire) for wire in observation_sites)
    z, zz = state.expectation_z_and_nearest_neighbor_zz(sites)
    return torch.cat((z, zz), dim=-1)


def predict(
    coupling: torch.Tensor,
    field: torch.Tensor,
    probe: Probe,
    *,
    observation_sites: Iterable[int],
    dt: float,
    max_bond: int,
    cutoff: float,
    device: torch.device | str,
) -> torch.Tensor:
    circuit = build_trotter_circuit(coupling, field, probe, dt=dt, device=device)
    state = fqb.run_mps(circuit, max_bond=max_bond, cutoff=cutoff)
    return local_observables(state, observation_sites).reshape(-1)


def predict_batch(
    coupling: torch.Tensor,
    field: torch.Tensor,
    probes: tuple[Probe, ...],
    *,
    observation_sites: Iterable[int],
    dt: float,
    max_bond: int,
    cutoff: float,
    device: torch.device | str,
    compiled_brickwork: bool = False,
    compiled_observables: bool = False,
) -> torch.Tensor:
    if compiled_brickwork:
        if cutoff != 0:
            raise ValueError("compiled brickwork currently requires cutoff=0")
        state = run_batched_brickwork_mps(
            coupling,
            field,
            [probe.flipped_sites for probe in probes],
            time_steps=probes[0].time_steps,
            dt=dt,
            max_bond=max_bond,
            compiled=True,
        )
        if compiled_observables:
            z, zz = compiled_local_z_zz(
                state, tuple(int(wire) for wire in observation_sites), compiled=True
            )
            return torch.cat((z, zz), dim=-1)
        return local_observables(state, observation_sites)
    else:
        circuit = build_batched_trotter_circuit(coupling, field, probes, dt=dt, device=device)
        state = fqb.run_mps(circuit, max_bond=max_bond, cutoff=cutoff)
        return local_observables(state, observation_sites)


def relative_parameter_error(estimate: torch.Tensor, truth: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(truth).clamp_min(torch.finfo(truth.dtype).eps)
    return float((torch.linalg.vector_norm(estimate - truth) / denominator).detach().cpu())
