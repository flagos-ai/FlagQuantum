# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_qft_demolition_measure.cc.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/qft/gen_qft_demolition_measure.cc

"""Source-ordered demolition Fourier measurement with exact phase feedback."""

from ...artifacts.phase_angle import PhaseAngle128
from ...artifacts.reversible_phase import (
    ReversiblePhaseOperation,
    ReversiblePhaseProgram,
)
from ..finite_field.polynomial import _validate_registers


def kickmix_qft_measurement_operations(
    target: tuple[int, ...], output: tuple[int, ...], *, inverse_qft: bool = False
) -> tuple[ReversiblePhaseOperation, ...]:
    """Measure a big-endian word in the source Fourier basis and clear it.

    Output slots form an equally wide big-endian classical word. The caller
    owns these slots, which are overwritten in source low-to-high order.
    No extra quantum work or classical scratch is needed. Empty logical
    words emit no operations and may be embedded in a positive-width circuit.
    Source zero-angle nodes below the fixed precision grid are retained.
    """
    if len(target) != len(output):
        raise ValueError(
            "Fourier measurement requires matching target and output widths"
        )
    if type(inverse_qft) is not bool:
        raise ValueError("Fourier measurement direction must be boolean")
    _validate_registers(target)
    if len(set(output)) != len(output) or any(
        type(bit) is not int or bit < 0 for bit in output
    ):
        raise ValueError(
            "Fourier measurement outputs must be distinct nonnegative slots"
        )
    word, bits = target[::-1], output[::-1]
    ops = [
        ReversiblePhaseOperation("swap", (word[i], word[-1 - i]))
        for i in range(len(word) // 2)
    ]
    for index, wire in enumerate(word):
        for previous in range(index):
            angle = PhaseAngle128.from_power_of_two_half_turns(previous - index)
            if inverse_qft:
                angle = -angle
            ops.append(
                ReversiblePhaseOperation(
                    "z_pow", (wire,), condition_bit=bits[previous], angle=angle
                )
            )
        ops.append(ReversiblePhaseOperation("hmr", (wire,), result_bit=bits[index]))
    return tuple(ops)


def kickmix_qft_measurement_program(
    target: tuple[int, ...],
    output: tuple[int, ...],
    *,
    n_qubits: int,
    inverse_qft: bool = False,
) -> ReversiblePhaseProgram:
    """Embed the logical operation while preserving unowned classical holes."""
    operations = kickmix_qft_measurement_operations(
        target, output, inverse_qft=inverse_qft
    )
    bits = max(output, default=-1) + 1
    inputs = tuple(bit for bit in range(bits) if bit not in output)
    return ReversiblePhaseProgram(n_qubits, bits, operations, inputs)
