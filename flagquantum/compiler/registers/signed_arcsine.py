# SPDX-License-Identifier: Apache-2.0
# Coherent sign restoration from Appendix D:
# https://arxiv.org/pdf/1805.12445#page=10

"""Signed wrapper for the Appendix D fixed-point arcsine."""

from ...core.ir import Instruction
from .bitwise import _validate_wires, fanout_ir, xor_ir
from .controlled_addition import (
    controlled_add_constant_adjoint_ir,
    controlled_add_constant_ir,
)
from .fixed_point_arcsine import (
    fixed_point_arcsine_ir,
    fixed_point_arcsine_requirements,
)


def signed_arcsine_requirements(
    source_fraction_bits: int,
    polynomial_fraction_bits: int,
    output_width: int,
    degree: int,
    *,
    precision: int,
    iterations: int,
) -> tuple[int, int]:
    """Return magnitude and angle storage, shared scratch and measurement slots."""
    work, slots = fixed_point_arcsine_requirements(
        source_fraction_bits,
        polynomial_fraction_bits,
        output_width - 1,
        degree,
        precision=precision,
        iterations=iterations,
    )
    magnitude_width = source_fraction_bits + 1
    negate_width = max(magnitude_width, output_width)
    return (
        magnitude_width + output_width + max(work, 2 * negate_width - 1),
        max(slots, negate_width - 1),
    )


def signed_arcsine_ir(
    source: tuple[int, ...],
    result: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    coefficients: tuple[int, ...],
    polynomial_fraction_bits: int,
    half_pi_word: int,
    precision: int,
    iterations: int,
    tuned: bool = False,
) -> tuple[Instruction, ...]:
    """XOR arcsine of x in [-1, 1] into a signed result, restoring all work.

    Source and result are big-endian two's-complement words with a sign
    bit and one integer magnitude bit. The half-pi constant and polynomial
    follow the unsigned kernel's domain and precision contract.
    """
    f, out = len(source) - 2, len(result)
    degree = len(coefficients) - 1
    config = {"precision": precision, "iterations": iterations}
    work, slots = signed_arcsine_requirements(
        f, polynomial_fraction_bits, out, degree, **config
    )
    if len(workspace) != work or len(classical_bits) != slots:
        raise ValueError(
            f"Signed arcsine requires {work} work qubits and {slots} slots"
        )
    if (
        any(type(bit) is not int or bit < 0 for bit in classical_bits)
        or len(set(classical_bits)) != slots
    ):
        raise ValueError("Signed arcsine requires distinct nonnegative slots")
    _validate_wires((*source, *result, *workspace))

    magnitude = workspace[: f + 1]
    angle = workspace[f + 1 : f + 1 + out]
    scratch = workspace[f + 1 + out :]
    sign = source[0]

    def negate(
        target: tuple[int, ...],
    ) -> tuple[tuple[Instruction, ...], tuple[Instruction, ...]]:
        width = len(target)
        invert = fanout_ir(sign, target)
        args = (
            sign,
            target,
            1,
            scratch[:width],
            scratch[width : 2 * width - 1],
            classical_bits[: width - 1],
        )
        return (
            (*invert, *controlled_add_constant_ir(*args)),
            (*controlled_add_constant_adjoint_ir(*args), *invert),
        )

    abs_forward, abs_reverse = negate(magnitude)
    sign_forward, sign_reverse = negate(angle)
    unsigned_work, unsigned_slots = fixed_point_arcsine_requirements(
        f, polynomial_fraction_bits, out - 1, degree, **config
    )
    evaluate = fixed_point_arcsine_ir(
        magnitude,
        angle[1:],
        scratch[:unsigned_work],
        classical_bits[:unsigned_slots],
        coefficients=coefficients,
        polynomial_fraction_bits=polynomial_fraction_bits,
        half_pi_word=half_pi_word,
        precision=precision,
        iterations=iterations,
        tuned=tuned,
    )
    copy_input = xor_ir(source[1:], magnitude)
    return (
        *copy_input,
        *abs_forward,
        *evaluate,
        *sign_forward,
        *xor_ir(angle, result),
        *sign_reverse,
        *evaluate,
        *abs_reverse,
        *copy_input,
    )
