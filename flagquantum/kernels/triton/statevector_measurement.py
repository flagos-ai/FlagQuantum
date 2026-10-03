"""Autograd-enabled statevector probability generation for CUDA tensors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from ._jit import jit
else:
    from triton import jit


class _ProbabilityContext(Protocol):
    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]: ...

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


class _PauliExpectationContext(Protocol):
    x_mask: int
    z_mask: int
    y_count: int

    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]: ...

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


class _MarginalProbabilityContext(Protocol):
    ordered_positions_packed: int
    sorted_positions_packed: int
    selected_count: int
    amplitude_count: int

    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]: ...

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


@jit
def _statevector_probabilities_kernel(
    state_parts: tl.tensor,
    probabilities: tl.tensor,
    element_count: tl.tensor,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < element_count
    real = tl.load(state_parts + 2 * offsets, mask=mask, other=0.0)
    imag = tl.load(state_parts + 2 * offsets + 1, mask=mask, other=0.0)
    tl.store(probabilities + offsets, real * real + imag * imag, mask=mask)


@jit
def _statevector_probabilities_backward_kernel(
    state_parts: tl.tensor,
    probability_gradient: tl.tensor,
    state_gradient_parts: tl.tensor,
    element_count: tl.tensor,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < element_count
    real = tl.load(state_parts + 2 * offsets, mask=mask, other=0.0)
    imag = tl.load(state_parts + 2 * offsets + 1, mask=mask, other=0.0)
    gradient = 2.0 * tl.load(
        probability_gradient + offsets,
        mask=mask,
        other=0.0,
    )
    tl.store(state_gradient_parts + 2 * offsets, gradient * real, mask=mask)
    tl.store(state_gradient_parts + 2 * offsets + 1, gradient * imag, mask=mask)


@jit
def _statevector_marginal_probability_partial_kernel(
    state_parts: tl.tensor,
    partials: tl.tensor,
    amplitude_count: tl.tensor,
    complement_count: tl.tensor,
    outcome_count: tl.tensor,
    chunk_count: tl.tensor,
    ordered_positions_packed: tl.tensor,
    sorted_positions_packed: tl.tensor,
    selected_count: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    programs_per_batch = outcome_count * chunk_count
    batch = program // programs_per_batch
    local_program = program % programs_per_batch
    outcome = local_program // chunk_count
    chunk = local_program % chunk_count
    complement = chunk * block_size + tl.arange(0, block_size)
    valid = complement < complement_count

    basis = complement
    for slot in range(selected_count):
        position = (sorted_positions_packed >> (5 * slot)) & 31
        lower_mask = (1 << position) - 1
        basis = (basis & lower_mask) | ((basis >> position) << (position + 1))
    for slot in range(selected_count):
        position = (ordered_positions_packed >> (5 * slot)) & 31
        outcome_bit = (outcome >> (selected_count - slot - 1)) & 1
        basis = basis | (outcome_bit << position)

    offsets = batch.to(tl.int64) * amplitude_count + basis
    real = tl.load(state_parts + 2 * offsets, mask=valid, other=0.0)
    imag = tl.load(state_parts + 2 * offsets + 1, mask=valid, other=0.0)
    partial = tl.sum(real * real + imag * imag, axis=0)
    tl.store(partials + program, partial)


@jit
def _statevector_marginal_probability_small_kernel(
    state_parts: tl.tensor,
    marginal: tl.tensor,
    amplitude_count: tl.tensor,
    ordered_positions_packed: tl.tensor,
    selected_count: tl.constexpr,
    outcome_count: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    basis = tl.arange(0, block_size)
    valid = basis < amplitude_count
    offsets = batch.to(tl.int64) * amplitude_count + basis
    real = tl.load(state_parts + 2 * offsets, mask=valid, other=0.0)
    imag = tl.load(state_parts + 2 * offsets + 1, mask=valid, other=0.0)
    probabilities = real * real + imag * imag
    projected = tl.zeros((block_size,), dtype=tl.int32)
    for slot in range(selected_count):
        position = (ordered_positions_packed >> (5 * slot)) & 31
        outcome_bit = (basis >> position) & 1
        projected = projected | (outcome_bit << (selected_count - slot - 1))
    for outcome in range(outcome_count):
        value = tl.sum(tl.where(projected == outcome, probabilities, 0.0), axis=0)
        tl.store(marginal + batch * outcome_count + outcome, value)


@jit
def _statevector_marginal_probability_backward_kernel(
    state_parts: tl.tensor,
    marginal_gradient: tl.tensor,
    state_gradient_parts: tl.tensor,
    amplitude_count: tl.tensor,
    element_count: tl.tensor,
    outcome_count: tl.tensor,
    ordered_positions_packed: tl.tensor,
    selected_count: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid = offsets < element_count
    batch = offsets // amplitude_count
    basis = offsets % amplitude_count
    outcome = tl.zeros((block_size,), dtype=tl.int32)
    for slot in range(selected_count):
        position = (ordered_positions_packed >> (5 * slot)) & 31
        outcome_bit = (basis >> position) & 1
        outcome = outcome | (outcome_bit << (selected_count - slot - 1))

    real = tl.load(state_parts + 2 * offsets, mask=valid, other=0.0)
    imag = tl.load(state_parts + 2 * offsets + 1, mask=valid, other=0.0)
    gradient = 2.0 * tl.load(
        marginal_gradient + batch * outcome_count + outcome,
        mask=valid,
        other=0.0,
    )
    tl.store(state_gradient_parts + 2 * offsets, gradient * real, mask=valid)
    tl.store(state_gradient_parts + 2 * offsets + 1, gradient * imag, mask=valid)


@jit
def _statevector_pauli_expectation_partial_kernel(
    state_parts: tl.tensor,
    partials: tl.tensor,
    amplitude_count: tl.tensor,
    chunk_count: tl.tensor,
    x_mask: tl.tensor,
    z_mask: tl.tensor,
    y_mod_four: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    batch = program // chunk_count
    chunk = program % chunk_count
    indices = chunk * block_size + tl.arange(0, block_size)
    valid = indices < amplitude_count
    partners = indices ^ x_mask
    batch_offset = batch.to(tl.int64) * amplitude_count
    state_offsets = batch_offset + indices
    partner_offsets = batch_offset + partners
    real = tl.load(state_parts + 2 * state_offsets, mask=valid, other=0.0)
    imag = tl.load(state_parts + 2 * state_offsets + 1, mask=valid, other=0.0)
    partner_real = tl.load(
        state_parts + 2 * partner_offsets,
        mask=valid,
        other=0.0,
    )
    partner_imag = tl.load(
        state_parts + 2 * partner_offsets + 1,
        mask=valid,
        other=0.0,
    )
    product_real = partner_real * real + partner_imag * imag
    product_imag = partner_real * imag - partner_imag * real
    if y_mod_four == 0:
        phased_real = product_real
    elif y_mod_four == 1:
        phased_real = -product_imag
    elif y_mod_four == 2:
        phased_real = -product_real
    else:
        phased_real = product_imag
    parity_bits = indices & z_mask
    parity_bits = parity_bits ^ (parity_bits >> 16)
    parity_bits = parity_bits ^ (parity_bits >> 8)
    parity_bits = parity_bits ^ (parity_bits >> 4)
    parity_bits = parity_bits ^ (parity_bits >> 2)
    parity_bits = parity_bits ^ (parity_bits >> 1)
    sign = 1.0 - 2.0 * (parity_bits & 1)
    partial = tl.sum(phased_real * sign, axis=0)
    tl.store(partials + batch * chunk_count + chunk, partial)


@jit
def _statevector_pauli_expectation_backward_kernel(
    state_parts: tl.tensor,
    expectation_gradient: tl.tensor,
    state_gradient_parts: tl.tensor,
    amplitude_count: tl.tensor,
    chunk_count: tl.tensor,
    x_mask: tl.tensor,
    z_mask: tl.tensor,
    y_mod_four: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    batch = program // chunk_count
    chunk = program % chunk_count
    offsets = chunk * block_size + tl.arange(0, block_size)
    valid = offsets < amplitude_count
    partners = offsets ^ x_mask
    batch_offset = batch.to(tl.int64) * amplitude_count
    partner_offsets = batch_offset + partners
    partner_real = tl.load(
        state_parts + 2 * partner_offsets,
        mask=valid,
        other=0.0,
    )
    partner_imag = tl.load(
        state_parts + 2 * partner_offsets + 1,
        mask=valid,
        other=0.0,
    )
    parity_bits = partners & z_mask
    parity_bits = parity_bits ^ (parity_bits >> 16)
    parity_bits = parity_bits ^ (parity_bits >> 8)
    parity_bits = parity_bits ^ (parity_bits >> 4)
    parity_bits = parity_bits ^ (parity_bits >> 2)
    parity_bits = parity_bits ^ (parity_bits >> 1)
    sign = 1.0 - 2.0 * (parity_bits & 1)
    if y_mod_four == 0:
        transformed_real = sign * partner_real
        transformed_imag = sign * partner_imag
    elif y_mod_four == 1:
        transformed_real = -sign * partner_imag
        transformed_imag = sign * partner_real
    elif y_mod_four == 2:
        transformed_real = -sign * partner_real
        transformed_imag = -sign * partner_imag
    else:
        transformed_real = sign * partner_imag
        transformed_imag = -sign * partner_real
    scale = 2.0 * tl.load(expectation_gradient + batch)
    output_offsets = batch_offset + offsets
    tl.store(
        state_gradient_parts + 2 * output_offsets,
        scale * transformed_real,
        mask=valid,
    )
    tl.store(
        state_gradient_parts + 2 * output_offsets + 1,
        scale * transformed_imag,
        mask=valid,
    )


def _reference(state: torch.Tensor) -> torch.Tensor:
    return torch.abs(state) ** 2


def _marginal_reference(
    state: torch.Tensor,
    qubits: tuple[int, ...],
    n_qubits: int,
) -> torch.Tensor:
    probabilities = torch.abs(state) ** 2
    shaped = probabilities.reshape((probabilities.shape[0],) + (2,) * n_qubits)
    selected = set(qubits)
    unselected_axes = tuple(
        wire + 1 for wire in range(n_qubits) if wire not in selected
    )
    marginal = shaped.sum(dim=unselected_axes) if unselected_axes else shaped
    current_order = tuple(sorted(qubits))
    if qubits != current_order:
        permutation = (0,) + tuple(current_order.index(qubit) + 1 for qubit in qubits)
        marginal = marginal.permute(permutation)
    return marginal.reshape(state.shape[0], 1 << len(qubits))


def _validate(state: torch.Tensor) -> None:
    if state.ndim != 2:
        raise ValueError("statevector probabilities require shape [batch, amplitudes]")
    if min(state.shape) <= 0:
        raise ValueError("statevector probability dimensions must be positive")
    if state.dtype not in (torch.complex64, torch.complex128):
        raise ValueError("statevector probabilities require a complex dtype")


def _supported(state: torch.Tensor) -> bool:
    return bool(
        state.is_cuda
        and state.dtype == torch.complex64
        and state.is_contiguous()
        and not state.is_conj()
        and not state.is_neg()
    )


def _normalize_marginal_qubits(
    qubits: Sequence[int],
    *,
    n_qubits: int,
) -> tuple[int, ...]:
    normalized: list[int] = []
    for index, qubit in enumerate(qubits):
        if isinstance(qubit, bool) or not isinstance(qubit, int):
            raise TypeError(f"Marginal qubit {index} must be an integer")
        if qubit < 0 or qubit >= n_qubits:
            raise ValueError(f"Marginal qubit {index} is outside the statevector")
        normalized.append(qubit)
    if len(set(normalized)) != len(normalized):
        raise ValueError("Marginal qubits must be unique")
    return tuple(normalized)


def _validate_marginal_state(state: torch.Tensor) -> int:
    _validate(state)
    amplitudes = int(state.shape[1])
    n_qubits = amplitudes.bit_length() - 1
    if 2**n_qubits != amplitudes:
        raise ValueError(
            "Marginal probabilities require a power-of-two amplitude count"
        )
    return n_qubits


def _pack_positions(positions: Sequence[int]) -> int:
    packed = 0
    for slot, position in enumerate(positions):
        packed |= position << (5 * slot)
    return packed


def _marginal_supported(
    state: torch.Tensor,
    *,
    n_qubits: int,
    selected_count: int,
) -> bool:
    return _supported(state) and n_qubits <= 30 and selected_count <= 8


def _normalize_pauli_product(
    operators: Sequence[tuple[int, str]],
    *,
    n_qubits: int,
) -> tuple[int, int, int]:
    seen: set[int] = set()
    x_mask = 0
    z_mask = 0
    y_count = 0
    for index, factor in enumerate(operators):
        if not isinstance(factor, tuple) or len(factor) != 2:
            raise TypeError(f"Pauli factor {index} must be a (wire, axis) tuple")
        wire, axis = factor
        if isinstance(wire, bool) or not isinstance(wire, int):
            raise TypeError(f"Pauli factor {index} wire must be an integer")
        if wire < 0 or wire >= n_qubits:
            raise ValueError(f"Pauli factor {index} wire is outside the statevector")
        if wire in seen:
            raise ValueError("Pauli-product wires must be unique")
        if not isinstance(axis, str):
            raise TypeError(f"Pauli factor {index} axis must be a string")
        normalized_axis = axis.lower()
        if normalized_axis not in {"x", "y", "z"}:
            raise ValueError("Pauli axes must be X, Y, or Z")
        seen.add(wire)
        bit = 1 << (n_qubits - wire - 1)
        if normalized_axis in {"x", "y"}:
            x_mask |= bit
        if normalized_axis in {"y", "z"}:
            z_mask |= bit
        if normalized_axis == "y":
            y_count += 1
    return x_mask, z_mask, y_count


def _validate_pauli_state(state: torch.Tensor) -> int:
    _validate(state)
    amplitudes = int(state.shape[1])
    n_wires = amplitudes.bit_length() - 1
    if 2**n_wires != amplitudes:
        raise ValueError("Pauli expectation requires a power-of-two amplitude count")
    return n_wires


def _pauli_expectation_reference(
    state: torch.Tensor,
    x_mask: int,
    z_mask: int,
    y_count: int,
) -> torch.Tensor:
    amplitude_count = int(state.shape[1])
    indices = torch.arange(amplitude_count, device=state.device)
    partners = indices ^ x_mask
    parity_bits = torch.bitwise_and(indices, z_mask)
    parity = torch.zeros_like(parity_bits)
    while z_mask:
        parity = torch.bitwise_xor(parity, parity_bits)
        parity_bits = torch.bitwise_right_shift(parity_bits, 1)
        z_mask >>= 1
    sign = 1 - 2 * torch.bitwise_and(parity, 1)
    product = torch.conj(state.index_select(1, partners)) * state
    phase = (1j) ** (y_count % 4)
    return torch.real(product * sign * phase).sum(dim=1)


def _pauli_supported(state: torch.Tensor) -> bool:
    return _supported(state) and int(state.shape[1]) <= 1 << 30


def _launch_forward(state: torch.Tensor) -> torch.Tensor:
    probabilities = torch.empty(
        state.shape,
        dtype=torch.float32,
        device=state.device,
    )
    element_count = state.numel()
    block_size = 256
    _statevector_probabilities_kernel[(triton.cdiv(element_count, block_size),)](
        torch.view_as_real(state),
        probabilities,
        element_count,
        block_size=block_size,
        num_warps=4,
        num_stages=2,
    )
    return probabilities


def _launch_backward(
    state: torch.Tensor,
    probability_gradient: torch.Tensor,
) -> torch.Tensor:
    probability_gradient = probability_gradient.contiguous()
    state_gradient = torch.empty_like(state)
    element_count = state.numel()
    block_size = 256
    _statevector_probabilities_backward_kernel[
        (triton.cdiv(element_count, block_size),)
    ](
        torch.view_as_real(state),
        probability_gradient,
        torch.view_as_real(state_gradient),
        element_count,
        block_size=block_size,
        num_warps=4,
        num_stages=2,
    )
    return state_gradient


class _StatevectorProbabilities(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _ProbabilityContext, state: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(state)
        return _launch_forward(state)

    @staticmethod
    def backward(
        ctx: _ProbabilityContext,
        probability_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        (state,) = ctx.saved_tensors
        return (_launch_backward(state, probability_gradient),)


def _launch_marginal_forward(
    state: torch.Tensor,
    ordered_positions_packed: int,
    sorted_positions_packed: int,
    selected_count: int,
) -> torch.Tensor:
    batch, amplitude_count = (int(value) for value in state.shape)
    outcome_count = 1 << selected_count
    complement_count = amplitude_count // outcome_count
    if amplitude_count <= 4096 and outcome_count <= 16:
        marginal = torch.empty(
            (batch, outcome_count),
            device=state.device,
            dtype=torch.float32,
        )
        block_size = 1 << (amplitude_count - 1).bit_length()
        _statevector_marginal_probability_small_kernel[(batch,)](
            torch.view_as_real(state),
            marginal,
            amplitude_count,
            ordered_positions_packed,
            selected_count=selected_count,
            outcome_count=outcome_count,
            block_size=block_size,
            num_warps=8,
            num_stages=2,
        )
        return marginal
    block_size = 1024
    chunk_count = triton.cdiv(complement_count, block_size)
    partials = torch.empty(
        (batch, outcome_count, chunk_count),
        device=state.device,
        dtype=torch.float32,
    )
    grid = batch * outcome_count * chunk_count
    _statevector_marginal_probability_partial_kernel[(grid,)](
        torch.view_as_real(state),
        partials,
        amplitude_count,
        complement_count,
        outcome_count,
        chunk_count,
        ordered_positions_packed,
        sorted_positions_packed,
        selected_count=selected_count,
        block_size=block_size,
        num_warps=8,
        num_stages=2,
    )
    if chunk_count == 1:
        return partials.squeeze(2)
    return partials.sum(dim=2)


def _launch_marginal_backward(
    state: torch.Tensor,
    marginal_gradient: torch.Tensor,
    ordered_positions_packed: int,
    selected_count: int,
) -> torch.Tensor:
    marginal_gradient = marginal_gradient.contiguous()
    state_gradient = torch.empty_like(state)
    amplitude_count = int(state.shape[1])
    element_count = state.numel()
    outcome_count = 1 << selected_count
    block_size = 256
    _statevector_marginal_probability_backward_kernel[
        (triton.cdiv(element_count, block_size),)
    ](
        torch.view_as_real(state),
        marginal_gradient,
        torch.view_as_real(state_gradient),
        amplitude_count,
        element_count,
        outcome_count,
        ordered_positions_packed,
        selected_count=selected_count,
        block_size=block_size,
        num_warps=4,
        num_stages=2,
    )
    return state_gradient


class _StatevectorMarginalProbabilities(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _MarginalProbabilityContext,
        state: torch.Tensor,
        ordered_positions_packed: int,
        sorted_positions_packed: int,
        selected_count: int,
    ) -> torch.Tensor:
        ctx.ordered_positions_packed = ordered_positions_packed
        ctx.sorted_positions_packed = sorted_positions_packed
        ctx.selected_count = selected_count
        ctx.amplitude_count = int(state.shape[1])
        ctx.save_for_backward(state)
        return _launch_marginal_forward(
            state,
            ordered_positions_packed,
            sorted_positions_packed,
            selected_count,
        )

    @staticmethod
    def backward(
        ctx: _MarginalProbabilityContext,
        marginal_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:
        (state,) = ctx.saved_tensors
        return (
            _launch_marginal_backward(
                state,
                marginal_gradient,
                ctx.ordered_positions_packed,
                ctx.selected_count,
            ),
            None,
            None,
            None,
        )


def _launch_pauli_forward(
    state: torch.Tensor,
    x_mask: int,
    z_mask: int,
    y_count: int,
) -> torch.Tensor:
    batch, amplitude_count = (int(value) for value in state.shape)
    block_size = 1024
    chunk_count = triton.cdiv(amplitude_count, block_size)
    partials = torch.empty(
        (batch, chunk_count),
        device=state.device,
        dtype=torch.float32,
    )
    _statevector_pauli_expectation_partial_kernel[(batch * chunk_count,)](
        torch.view_as_real(state),
        partials,
        amplitude_count,
        chunk_count,
        x_mask,
        z_mask,
        y_mod_four=y_count % 4,
        block_size=block_size,
        num_warps=8,
        num_stages=2,
    )
    return partials.sum(dim=1)


def _launch_pauli_backward(
    state: torch.Tensor,
    expectation_gradient: torch.Tensor,
    x_mask: int,
    z_mask: int,
    y_count: int,
) -> torch.Tensor:
    expectation_gradient = expectation_gradient.contiguous()
    state_gradient = torch.empty_like(state)
    batch, amplitude_count = (int(value) for value in state.shape)
    block_size = 256
    chunk_count = triton.cdiv(amplitude_count, block_size)
    _statevector_pauli_expectation_backward_kernel[(batch * chunk_count,)](
        torch.view_as_real(state),
        expectation_gradient,
        torch.view_as_real(state_gradient),
        amplitude_count,
        chunk_count,
        x_mask,
        z_mask,
        y_mod_four=y_count % 4,
        block_size=block_size,
        num_warps=4,
        num_stages=2,
    )
    return state_gradient


class _StatevectorPauliExpectation(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _PauliExpectationContext,
        state: torch.Tensor,
        x_mask: int,
        z_mask: int,
        y_count: int,
    ) -> torch.Tensor:
        ctx.x_mask = x_mask
        ctx.z_mask = z_mask
        ctx.y_count = y_count
        ctx.save_for_backward(state)
        return _launch_pauli_forward(state, x_mask, z_mask, y_count)

    @staticmethod
    def backward(
        ctx: _PauliExpectationContext,
        expectation_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:
        (state,) = ctx.saved_tensors
        return (
            _launch_pauli_backward(
                state,
                expectation_gradient,
                ctx.x_mask,
                ctx.z_mask,
                ctx.y_count,
            ),
            None,
            None,
            None,
        )


def statevector_probabilities(state: torch.Tensor) -> torch.Tensor:
    """Return one probability per amplitude of a batched flat statevector.

    The CUDA complex64 path fuses real/imaginary magnitude work into one Triton
    pass and provides an explicit first-order backward kernel. Unsupported
    devices, dtypes, or layouts retain the exact PyTorch operation.
    """

    _validate(state)
    if not _supported(state):
        return _reference(state)
    apply: Callable[[torch.Tensor], object] = _StatevectorProbabilities.apply
    result = apply(state)
    if not isinstance(result, torch.Tensor):
        raise TypeError("Statevector probability autograd must return a tensor")
    return result


def statevector_marginal_probabilities(
    state: torch.Tensor,
    qubits: Sequence[int],
) -> torch.Tensor:
    """Return the joint marginal probability tensor for selected qubits.

    Wire zero addresses the most-significant statevector bit, and output bits
    follow the exact order supplied in ``qubits``. The CUDA complex64 path fuses
    magnitude generation, basis projection, and complement reduction. Inputs
    outside its bounded support matrix retain an exact differentiable PyTorch
    implementation.
    """

    n_qubits = _validate_marginal_state(state)
    normalized_qubits = _normalize_marginal_qubits(qubits, n_qubits=n_qubits)
    if not _marginal_supported(
        state,
        n_qubits=n_qubits,
        selected_count=len(normalized_qubits),
    ):
        return _marginal_reference(state, normalized_qubits, n_qubits)
    ordered_positions = tuple(n_qubits - qubit - 1 for qubit in normalized_qubits)
    ordered_positions_packed = _pack_positions(ordered_positions)
    sorted_positions_packed = _pack_positions(tuple(sorted(ordered_positions)))
    apply: Callable[[torch.Tensor, int, int, int], object] = (
        _StatevectorMarginalProbabilities.apply
    )
    result = apply(
        state,
        ordered_positions_packed,
        sorted_positions_packed,
        len(normalized_qubits),
    )
    if not isinstance(result, torch.Tensor):
        raise TypeError(
            "Statevector marginal probability autograd must return a tensor"
        )
    return result


def statevector_pauli_expectation(
    state: torch.Tensor,
    operators: Sequence[tuple[int, str]],
) -> torch.Tensor:
    """Return one exact Pauli-product expectation per flat statevector batch.

    Wire zero addresses the most-significant statevector bit, matching the
    simulation package. The CUDA complex64 path fuses Pauli permutation, phase,
    and inner-product work. Unsupported devices, dtypes, or layouts retain an
    exact differentiable PyTorch implementation.
    """

    n_qubits = _validate_pauli_state(state)
    x_mask, z_mask, y_count = _normalize_pauli_product(
        operators,
        n_qubits=n_qubits,
    )
    if not _pauli_supported(state):
        return _pauli_expectation_reference(state, x_mask, z_mask, y_count)
    apply: Callable[[torch.Tensor, int, int, int], object] = (
        _StatevectorPauliExpectation.apply
    )
    result = apply(state, x_mask, z_mask, y_count)
    if not isinstance(result, torch.Tensor):
        raise TypeError("Statevector Pauli expectation autograd must return a tensor")
    return result


__all__ = [
    "statevector_marginal_probabilities",
    "statevector_pauli_expectation",
    "statevector_probabilities",
]
