"""Low-rank product formulas built from native one- and two-qubit gates."""

from __future__ import annotations

import torch

from ...circuit import Circuit
from ...core.algebra.pauli import count, scalar
from .low_rank import LowRankFactors
from .lowering import append_orbital_rotation
from .numerics import static_tensor


def _time_tensor(time: torch.Tensor | float) -> torch.Tensor:
    if isinstance(time, torch.Tensor):
        if (
            time.ndim != 0
            or time.is_complex()
            or time.dtype == torch.bool
            or not time.is_floating_point()
            or not bool(torch.isfinite(time))
        ):
            raise ValueError("time must be a finite real floating scalar tensor")
        return time.to(dtype=torch.float64)
    numeric = scalar(time)
    if numeric.imag:
        raise ValueError("time must be real")
    return torch.tensor(numeric.real, dtype=torch.float64)


def _number_phase(circuit: Circuit, qubit: int, angle: torch.Tensor) -> None:
    phase = torch.exp(-1j * angle)
    circuit.any(
        qubit,
        matrix=torch.diag(torch.stack((torch.ones_like(phase), phase))),
        name="density_phase",
    )


def _density_evolution(
    circuit: Circuit,
    diagonal: torch.Tensor,
    time: torch.Tensor,
    *,
    squared: bool,
    coefficient: float = 1,
) -> None:
    values = diagonal.to(device=time.device)
    for qubit, value in enumerate(values):
        _number_phase(
            circuit, qubit, time * coefficient * (value.square() if squared else value)
        )
    if squared:
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                phase = torch.exp(
                    -2j * time * coefficient * values[left] * values[right]
                )
                one = torch.ones_like(phase)
                circuit.any(
                    left,
                    right,
                    matrix=torch.diag(torch.stack((one, one, one, phase))),
                    name="density_pair_phase",
                )


def low_rank_evolution(
    one_body: torch.Tensor,
    factors: LowRankFactors,
    time: torch.Tensor | float,
    *,
    constant: float = 0,
    steps: int = 1,
    order: int = 2,
) -> Circuit:
    """Approximate exp(-i time H) for spin-interleaved real spatial integrals.

    Adds factors.correction to one_body exactly once. The constant (e.g. nuclear
    plus frozen-core energy) contributes an exact global phase. Time tensors
    remain differentiable; static integrals and rotations refuse autograd.
    Product-formula error is separate from factors.discarded_bound. Output
    layout equals input JW layout; density interactions may be nonadjacent.
    """
    matrix = static_tensor(one_body, "one_body", ndim=2)
    n = factors.n_orbitals
    if matrix.shape != (n, n) or not torch.allclose(
        matrix, matrix.mH, atol=1e-10, rtol=1e-10
    ):
        raise ValueError("one_body must be Hermitian and match the factor orbitals")
    repetitions = count(steps, "steps", minimum=1)
    formula_order = count(order, "order", minimum=1)
    if formula_order not in (1, 2):
        raise ValueError("product formula order must be 1 or 2")
    offset = scalar(constant)
    if offset.imag:
        raise ValueError("constant must be real")
    duration = _time_tensor(time)
    circuit = Circuit(2 * n, dtype=torch.complex128, device=duration.device)
    phase = torch.exp(-1j * duration * offset.real)
    circuit.any(
        0,
        matrix=phase * torch.eye(2, dtype=torch.complex128, device=duration.device),
        name="energy_offset_phase",
    )
    # Each entry is (eigenbasis, eigenvalues, squared-density coefficient).
    energies, orbitals = torch.linalg.eigh(matrix + factors.correction)
    components: list[tuple[torch.Tensor, torch.Tensor, float | None]] = [
        (orbitals, energies, None)
    ]
    for weight, factor in zip(factors.eigenvalues, factors.factors, strict=True):
        if float(weight) == 0:
            continue
        values, basis = torch.linalg.eigh(factor)
        components.append((basis.to(torch.complex128), values, 0.5 * float(weight)))
    schedule = (
        [(component, 1.0) for component in components]
        if formula_order == 1
        else [(component, 0.5) for component in components]
        + [(component, 0.5) for component in reversed(components)]
    )
    for _ in range(repetitions):
        for (basis, values, coefficient), fraction in schedule:
            spin_basis = torch.kron(basis.contiguous(), torch.eye(2, dtype=basis.dtype))
            append_orbital_rotation(circuit, spin_basis.mH.contiguous())
            _density_evolution(
                circuit,
                values.repeat_interleave(2),
                duration * (fraction / repetitions),
                squared=coefficient is not None,
                coefficient=1 if coefficient is None else coefficient,
            )
            append_orbital_rotation(circuit, spin_basis)
    return circuit
