# SPDX-License-Identifier: Apache-2.0
# Seed and Newton iteration from Appendix C:
# https://arxiv.org/pdf/1805.12445#page=8

"""Compose a clean fixed-point inverse square root from C1 and Newton steps."""

from collections.abc import Callable

from ...core.ir import Instruction
from .bitwise import _validate_wires, xor_ir
from .inverse_sqrt_newton import (
    inverse_sqrt_newton_requirements,
    inverse_sqrt_newton_step_ir,
)
from .inverse_sqrt_seed import inverse_sqrt_seed_ir, inverse_sqrt_seed_requirements


def _inverse_sqrt_layout(
    source_width: int, source_fraction_bits: int, precision: int, iterations: int
) -> tuple[int, tuple[int, int], tuple[int, int]]:
    seed = inverse_sqrt_seed_requirements(
        source_width,
        source_fraction_bits=source_fraction_bits,
        result_fraction_bits=precision,
    )
    if type(iterations) is not int or iterations < 0:
        raise ValueError("Inverse square root requires a nonnegative iteration count")
    smallest_shift = (source_fraction_bits - source_width + 1) // 2
    if precision < 1 - smallest_shift:
        raise ValueError(
            "Inverse square root precision must keep positive seeds nonzero"
        )
    estimate_width = precision + source_fraction_bits // 2 + 1
    step = (
        inverse_sqrt_newton_requirements(
            source_width,
            estimate_width,
            source_fraction_bits=source_fraction_bits,
            estimate_fraction_bits=precision,
        )
        if iterations
        else (0, 0)
    )
    return estimate_width, seed, step


def inverse_sqrt_requirements(
    source_width: int, *, source_fraction_bits: int, precision: int, iterations: int
) -> tuple[int, int]:
    """Return clean history plus scratch width and reusable slot count."""
    estimate_width, (seed_work, seed_slots), (step_work, step_slots) = (
        _inverse_sqrt_layout(source_width, source_fraction_bits, precision, iterations)
    )
    return (
        (iterations + 1) * estimate_width + max(seed_work, step_work),
        max(seed_slots, step_slots),
    )


def inverse_sqrt_ir(
    source: tuple[int, ...],
    result: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    source_fraction_bits: int,
    precision: int,
    iterations: int,
) -> tuple[Instruction, ...]:
    """XOR the C1/Newton estimate into a big-endian result and erase history.

    Positive inputs give a lower estimate of the inverse square root. Zero
    gives a zero mask. Keep floor(source_fraction_bits/2)+1 integer bits
    and optionally shorten the fractional suffix. The caller provides clean
    workspace and reusable classical slots.
    """
    estimate_width, (seed_work, seed_slots), (step_work, step_slots) = (
        _inverse_sqrt_layout(len(source), source_fraction_bits, precision, iterations)
    )
    if not estimate_width - precision <= len(result) <= estimate_width:
        raise ValueError("Inverse square root result must keep all integer bits")
    history_width = (iterations + 1) * estimate_width
    work = history_width + max(seed_work, step_work)
    slots = max(seed_slots, step_slots)
    if len(workspace) != work or len(classical_bits) != slots:
        raise ValueError(
            f"Inverse square root requires {work} work wires and {slots} slots"
        )
    if (
        any(type(bit) is not int or bit < 0 for bit in classical_bits)
        or len(set(classical_bits)) != slots
    ):
        raise ValueError("Inverse square root requires distinct nonnegative slots")
    _validate_wires((*source, *result, *workspace))

    return _with_inverse_sqrt_estimate_ir(
        source,
        workspace,
        classical_bits,
        source_fraction_bits=source_fraction_bits,
        precision=precision,
        iterations=iterations,
        consume=lambda estimate: xor_ir(estimate[: len(result)], result),
    )


def _with_inverse_sqrt_estimate_ir(
    source: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    source_fraction_bits: int,
    precision: int,
    iterations: int,
    consume: Callable[[tuple[int, ...]], tuple[Instruction, ...]],
) -> tuple[Instruction, ...]:
    """Use the last estimate before uncomputing history and with scratch clean."""
    estimate_width, (seed_work, seed_slots), (step_work, step_slots) = (
        _inverse_sqrt_layout(len(source), source_fraction_bits, precision, iterations)
    )
    history_width = (iterations + 1) * estimate_width
    history = tuple(
        workspace[i : i + estimate_width]
        for i in range(0, history_width, estimate_width)
    )
    scratch = workspace[history_width:]
    seed = inverse_sqrt_seed_ir(
        source,
        history[0],
        scratch[:seed_work],
        classical_bits[:seed_slots],
        source_fraction_bits=source_fraction_bits,
        result_fraction_bits=precision,
    )
    updates = tuple(
        inverse_sqrt_newton_step_ir(
            source,
            previous,
            following,
            scratch[:step_work],
            classical_bits[:step_slots],
            source_fraction_bits=source_fraction_bits,
            estimate_fraction_bits=precision,
        )
        for previous, following in zip(history[:-1], history[1:], strict=True)
    )
    return (
        *seed,
        *(gate for update in updates for gate in update),
        *consume(history[-1]),
        *(gate for update in reversed(updates) for gate in update),
        *seed,
    )
