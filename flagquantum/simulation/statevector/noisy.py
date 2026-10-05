"""Numerical kernels for batched statevector quantum trajectories.

**Why a channel's operators decide which kernel runs it.** A channel is either a
probability distribution over unitary branches or it is not, and which one it is
follows from the operators: every Kraus operator is a non-negative real scale
times a unitary exactly when the channel is a mixture. A mixture can be sampled
once per trajectory row against the channel's own weights and the drawn branch
applied as a unitary, so no branch state is materialized to measure weights from,
and only the branches a draw actually selected are applied. A channel that is not
a mixture has no such reading, and its branch weights only exist relative to the
state being evolved, so that route has to apply every operator to the state before
it can normalize. The two are different estimators over the same channel only in
cost: for a mixture the state-dependent weights the generic route measures are the
channel's weights, which is what :func:`run_noisy_trajectory_batch` reports as the
mixture and the generic route agreeing, and on this box the two routes return
bit-identical states at the same seeds.

**What is not here.** This module reads the classification and does not define
it: :attr:`flagquantum.noise.KrausChannel.unitary_mixture` owns the verdict, and
the same answer decides what the stabilizer engine can track. A channel that is
not a mixture is not thereby refused -- it reaches the generic route, or the
amplitude-damping kernel when it is that channel at one wire. Amplitude damping
is deliberately still selected by its opcode rather than by the structure of its
operators, because that kernel's saving comes from the triangular form of the
jump operator rather than from a mixture, and no structural test for it is stated
here.
"""

from __future__ import annotations

import torch

from ...core.ir import CircuitIR
from ...noise import KrausChannel, UnitaryMixture
from ..gate_matrix import gate_matrix
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


def _branch_mixture(
    name: str, operators: tuple[torch.Tensor, ...]
) -> UnitaryMixture | None:
    """Return this channel instruction's branches, or ``None`` if it has none.

    The question is asked of the operators, so a channel reaches the branch
    route because of what it holds rather than because of what it is called. The
    verdict itself belongs to :attr:`KrausChannel.unitary_mixture`, and asking
    there is what keeps this route and the stabilizer engine from answering the
    same question twice with two rules.

    An operator set that is not a trace-preserving channel is left to the routes
    that already carry one rather than becoming a new refusal here: this kernel
    is not where a malformed channel is judged, and a set that does not define a
    channel has no mixture to report either way.
    """

    try:
        channel = KrausChannel(name, operators)
    except ValueError:
        return None
    return channel.unitary_mixture


def run_noisy_trajectory_batch(
    initial: torch.Tensor,
    ir: CircuitIR,
    generators: list[torch.Generator],
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """Evolve one batch of already-lowered noisy trajectories.

    The third count is the events that took the branch route and the fifth the
    events that took the generic Kraus route; they are separate because they are
    the two different estimators over a channel described in the module
    docstring, and a caller reading only the result cannot otherwise tell which
    one produced it.
    """

    circuit_batch = initial.shape[0]
    state = initial.unsqueeze(0).expand(len(generators), -1, -1).clone()
    unitary_events = 0
    amplitude_events = 0
    generic_events = 0
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
        mixture = _branch_mixture(instruction.name, operators)
        if mixture is not None:
            # One draw per trajectory row against the channel's own weights, and
            # one application per branch that draw actually selected, so the
            # sixteen-branch two-qubit channel pays for the branches it drew
            # rather than for all sixteen. Applying every branch and gathering
            # the drawn one is the same estimator at a higher cost, which is why
            # the drawn branch is the one that is applied.
            probabilities = mixture.probabilities.to(dtype=state.real.dtype).expand(
                state.shape[0], state.shape[1], -1
            )
            choices = sample_rows(probabilities, generators)
            # ``mixture`` carries no zero-weight branch -- the classifier drops an
            # operator of scale zero rather than offering it -- so every row draws
            # exactly one index and every row is written exactly once. The base
            # value of ``selected`` is therefore never read, and it is the state
            # being evolved rather than a copy of it.
            selected = state
            for index, unitary in enumerate(mixture.unitaries):
                rows = (choices == index)[..., None]
                if not bool(rows.any()):
                    continue
                selected = torch.where(
                    rows,
                    apply_matrix_batched(state, unitary, instruction.wires, ir.n_wires),
                    selected,
                )
            state = selected
            unitary_events += state.shape[0] * circuit_batch
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
        unitary_events,
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
