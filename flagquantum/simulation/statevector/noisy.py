"""Numerical kernels for batched statevector quantum trajectories."""

from __future__ import annotations

import torch

from ...core.ir import CircuitIR
from ..gate_matrix import gate_matrix
from ..matrices import GATE_MAT_DICT
from .operations import _environment_flag


def apply_matrix_batched(
    state: torch.Tensor,
    matrix: torch.Tensor,
    wires: tuple[int, ...],
    n_wires: int,
) -> torch.Tensor:
    """Apply an unbatched or circuit-batched matrix to trajectory states."""

    trajectories, circuit_batch, _ = state.shape
    width = len(wires)
    dimension = 2**width
    other_wires = tuple(wire for wire in range(n_wires) if wire not in wires)
    permutation = (0, 1) + tuple(wire + 2 for wire in wires + other_wires)
    inverse = [0] * (n_wires + 2)
    for destination, source in enumerate(permutation):
        inverse[source] = destination
    flat = (
        state.reshape(trajectories, circuit_batch, *((2,) * n_wires))
        .permute(permutation)
        .reshape(trajectories * circuit_batch, dimension, -1)
    )
    matrix = torch.as_tensor(matrix, device=state.device, dtype=state.dtype)
    if matrix.ndim == 2:
        applied = torch.matmul(matrix, flat)
    elif matrix.ndim == 3 and matrix.shape[0] == circuit_batch:
        expanded = matrix.unsqueeze(0).expand(trajectories, -1, -1, -1)
        applied = torch.bmm(expanded.reshape(-1, dimension, dimension), flat)
    else:
        raise ValueError(
            "gate matrix must be unbatched or match the circuit batch dimension"
        )
    return (
        applied.reshape(trajectories, circuit_batch, *((2,) * n_wires))
        .permute(tuple(inverse))
        .reshape(trajectories, circuit_batch, -1)
    )


def _vectorized_row_sampling_enabled() -> bool:
    """Whether a categorical row is sampled by inverting its cumulative sum.

    Off by default, and unlike the other switches in this package that is not a
    matter of the last bit: ``torch.multinomial`` and a uniform inverted against
    a cumulative sum are different estimators over the same distribution, so the
    same generator does not produce the same choice. The random streams are
    derived from ``(seed, trajectory_id)`` and callers replay seeded runs, so the
    mapping is part of the observable contract and a caller opts in deliberately.
    """

    return _environment_flag("FQ_CPU_VECTORIZED_ROW_SAMPLING", default=False)


def _sample_rows_per_row(
    probabilities: torch.Tensor, generators: list[torch.Generator]
) -> torch.Tensor:
    """One ``torch.multinomial`` call per trajectory row, the shipped path."""

    choices = [
        torch.multinomial(row, 1, replacement=True, generator=generator).squeeze(-1)
        for row, generator in zip(probabilities, generators, strict=True)
    ]
    return torch.stack(choices)


def _sample_rows_by_inverse_cdf(
    probabilities: torch.Tensor, generators: list[torch.Generator]
) -> torch.Tensor:
    """Draw one uniform per row and invert it against one batched cumulative sum.

    The result has the same shape and the same per-trajectory stream as the
    per-row path: a trajectory's row is still drawn entirely from the generator
    its id derives, so the trajectory batch size keeps not changing a seeded
    result. What is batched is only the search, so ``zip(..., strict=True)``
    keeps a mismatched generator list reporting the same ``ValueError``.

    A row that ``torch.multinomial`` would refuse is handed to the per-row path
    rather than answered here: a non-positive total, or an entry that is
    negative, ``nan`` or ``inf``. Inverting a cumulative sum has no opinion about
    any of those - it returns an index - so answering would replace a raised error
    with a silently selected branch. The check is one pass over the request and
    costs what the ``cumsum`` next to it costs.
    """

    uniforms = torch.stack(
        [
            torch.rand(row.shape[0], generator=generator, dtype=probabilities.dtype)
            for row, generator in zip(probabilities, generators, strict=True)
        ]
    )
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    cumulative = torch.cumsum(flat, dim=-1)
    totals = cumulative[..., -1:]
    well_formed = bool(torch.all((flat >= 0) & torch.isfinite(flat)))
    if not well_formed or not bool(torch.all(totals > 0)):
        return _sample_rows_per_row(probabilities, generators)
    chosen = torch.searchsorted(cumulative, uniforms.reshape(-1, 1) * totals)
    return chosen.reshape(uniforms.shape)


def sample_rows(
    probabilities: torch.Tensor,
    generators: list[torch.Generator],
) -> torch.Tensor:
    """Sample one categorical choice per trajectory row."""

    if _vectorized_row_sampling_enabled():
        return _sample_rows_by_inverse_cdf(probabilities, generators)
    return _sample_rows_per_row(probabilities, generators)


def apply_kraus_batched(
    state: torch.Tensor,
    operators: tuple[torch.Tensor, ...],
    wires: tuple[int, ...],
    n_wires: int,
    generators: list[torch.Generator],
) -> torch.Tensor:
    """Sample and normalize a generic Kraus branch for each trajectory."""

    branches = torch.stack(
        [
            apply_matrix_batched(state, operator, wires, n_wires)
            for operator in operators
        ],
        dim=2,
    )
    probabilities = torch.sum(torch.abs(branches) ** 2, dim=-1).real
    probabilities = torch.clamp(probabilities, min=0)
    totals = probabilities.sum(dim=-1, keepdim=True)
    if bool(torch.any(totals <= 0)):
        raise RuntimeError("Kraus channel produced zero total branch probability")
    choices = sample_rows(probabilities / totals, generators)
    selected = torch.gather(
        branches,
        2,
        choices[..., None, None].expand(-1, -1, 1, branches.shape[-1]),
    ).squeeze(2)
    selected_probability = torch.gather(probabilities, 2, choices[..., None]).squeeze(
        -1
    )
    return (
        selected / torch.sqrt(torch.clamp(selected_probability, min=1e-30))[..., None]
    )


def apply_amplitude_damping_batched(
    state: torch.Tensor,
    operators: tuple[torch.Tensor, ...],
    wire: int,
    n_wires: int,
    generators: list[torch.Generator],
) -> torch.Tensor:
    """Sample amplitude damping without materializing all branch states."""

    gamma = torch.abs(operators[1][0, 1]) ** 2
    trajectories, circuit_batch, _ = state.shape
    other_wires = tuple(index for index in range(n_wires) if index != wire)
    permutation = (0, 1, wire + 2) + tuple(index + 2 for index in other_wires)
    inverse = [0] * (n_wires + 2)
    for destination, source in enumerate(permutation):
        inverse[source] = destination
    packed = (
        state.reshape(trajectories, circuit_batch, *((2,) * n_wires))
        .permute(permutation)
        .reshape(trajectories, circuit_batch, 2, -1)
    )
    zero, one = packed[:, :, 0], packed[:, :, 1]
    jump_probability = torch.clamp(
        gamma.real * torch.sum(torch.abs(one) ** 2, dim=-1), min=0, max=1
    )
    probabilities = torch.stack((1 - jump_probability, jump_probability), dim=-1)
    choices = sample_rows(probabilities, generators)
    jump = choices.bool()[..., None]
    out_zero = torch.where(jump, torch.sqrt(gamma) * one, zero)
    out_one = torch.where(jump, torch.zeros_like(one), torch.sqrt(1 - gamma) * one)
    selected_probability = torch.where(
        jump[..., 0], jump_probability, 1 - jump_probability
    )
    packed_out = (
        torch.stack((out_zero, out_one), dim=2)
        / torch.sqrt(torch.clamp(selected_probability, min=1e-30))[..., None, None]
    )
    return (
        packed_out.reshape(trajectories, circuit_batch, *((2,) * n_wires))
        .permute(tuple(inverse))
        .reshape_as(state)
    )


def expectation_z(state: torch.Tensor, n_wires: int) -> torch.Tensor:
    """Return per-trajectory Z expectations for every wire."""

    indices = torch.arange(state.shape[-1], device=state.device)
    shifts = torch.arange(n_wires - 1, -1, -1, device=state.device)
    signs = 1 - 2 * ((indices[:, None] >> shifts) & 1)
    return torch.matmul(torch.abs(state) ** 2, signs.to(dtype=state.real.dtype))


def run_noisy_trajectory_batch(
    initial: torch.Tensor,
    ir: CircuitIR,
    generators: list[torch.Generator],
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """Evolve one batch of already-lowered noisy trajectories."""

    circuit_batch = initial.shape[0]
    state = initial.unsqueeze(0).expand(len(generators), -1, -1).clone()
    pauli_events = 0
    amplitude_events = 0
    generic_events = 0
    pauli_matrices = {
        name: torch.as_tensor(matrix, device=initial.device, dtype=initial.dtype)
        for name, matrix in GATE_MAT_DICT.items()
        if name in {"i", "x", "y", "z"}
    }
    for instruction in ir.instructions:
        if not instruction.metadata.get("is_channel"):
            matrix = gate_matrix(
                instruction,
                bsz=circuit_batch,
                device=initial.device,
                dtype=initial.dtype,
            )
            state = apply_matrix_batched(state, matrix, instruction.wires, ir.n_wires)
            continue
        if instruction.matrix is None:
            raise ValueError(
                f"Noise channel {instruction.name!r} requires Kraus matrices."
            )
        operators = tuple(
            torch.as_tensor(operator, device=initial.device, dtype=initial.dtype)
            for operator in instruction.matrix
        )
        if instruction.name in {"bit_flip", "phase_flip", "depolarizing"}:
            probabilities = torch.tensor(
                [
                    float((torch.real(torch.trace(op.mH @ op)) / 2).item())
                    for op in operators
                ],
                device=state.device,
                dtype=state.real.dtype,
            ).expand(state.shape[0], state.shape[1], -1)
            choices = sample_rows(probabilities, generators)
            names = (
                ("i", "x")
                if instruction.name == "bit_flip"
                else (
                    ("i", "z")
                    if instruction.name == "phase_flip"
                    else ("i", "x", "y", "z")
                )
            )
            branches = torch.stack(
                [
                    apply_matrix_batched(
                        state, pauli_matrices[name], instruction.wires, ir.n_wires
                    )
                    for name in names
                ],
                dim=2,
            )
            state = torch.gather(
                branches,
                2,
                choices[..., None, None].expand(-1, -1, 1, state.shape[-1]),
            ).squeeze(2)
            pauli_events += state.shape[0] * circuit_batch
        elif instruction.name == "amplitude_damping" and len(instruction.wires) == 1:
            state = apply_amplitude_damping_batched(
                state, operators, instruction.wires[0], ir.n_wires, generators
            )
            amplitude_events += state.shape[0] * circuit_batch
        else:
            state = apply_kraus_batched(
                state, operators, instruction.wires, ir.n_wires, generators
            )
            generic_events += state.shape[0] * circuit_batch
    return (
        state,
        expectation_z(state, ir.n_wires),
        pauli_events,
        amplitude_events,
        generic_events,
    )


__all__ = (
    "apply_amplitude_damping_batched",
    "apply_kraus_batched",
    "apply_matrix_batched",
    "expectation_z",
    "run_noisy_trajectory_batch",
    "sample_rows",
)
