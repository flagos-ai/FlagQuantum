# SPDX-License-Identifier: Apache-2.0
# Qualtran v0.7.0 InvertRealNumber models the initial estimate.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/multiplication.py
# Executable construction: https://arxiv.org/pdf/1511.08253#page=10

"""Power-of-two starting estimate for fixed-point reciprocal iteration."""

from ....core.ir import Instruction
from ..bits.bitwise import _validate_wires
from ..control.multi_control import multi_control_x_ir


def reciprocal_seed_ir(
    source: tuple[int, ...],
    seed: tuple[int, ...],
    flag: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    source_fraction_bits: int,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Compute 2**(-p) for p equal to the integer part's bit length.

    Source and seed are big endian, with the seed entirely fractional.
    Supply one clean flag and one fewer cleanup work qubit than integer
    input bits. Forward requires a zero seed. The adjoint requires the
    computed seed. Both restore flag and work. A zero integer part yields
    a zero seed.
    """
    if type(source_fraction_bits) is not int or not 0 <= source_fraction_bits < len(
        source
    ):
        raise ValueError("Reciprocal seed requires a nonempty integer part")
    if type(adjoint) is not bool:
        raise ValueError("Reciprocal seed adjoint must be boolean")
    _validate_wires((*source, *seed, flag, *workspace))
    width = len(source) - source_fraction_bits
    if len(seed) < width:
        raise ValueError("Reciprocal seed needs a fractional wire for each integer bit")

    targets = seed[:width]
    select = [
        Instruction("cx", (source[0], flag)),
        Instruction("cx", (source[0], targets[-1])),
    ]
    for i, control in enumerate(source[1:width], start=1):
        target = targets[-1 - i]
        toggle = Instruction("x", (flag,))
        select.extend(
            (
                toggle,
                Instruction("ccx", (control, flag, target)),
                toggle,
                Instruction("cx", (target, flag)),
            )
        )
    clear = multi_control_x_ir(
        targets,
        (0,) * width,
        flag,
        workspace,
        classical_bits,
    )
    finish = Instruction("x", (flag,))
    if adjoint:
        return (finish, *clear, *reversed(select))
    return (*select, *clear, finish)
