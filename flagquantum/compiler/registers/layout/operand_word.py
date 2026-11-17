# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix operand-array conversion.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/py/val/converted_array.pybind.cc

"""Bind typed operands and integer words in native big-endian order."""

from collections.abc import Iterable, Sized
from itertools import islice
from typing import cast

from ..primitives._mixed_gates import _Control, _XBasisTarget
from ..primitives.operand_binding import _Operand, bind_operand
from .register_bits import split_integer


def bind_operand_word(
    value: object, *, sequence_only: bool = False
) -> tuple[_Operand, ...]:
    """Snapshot ordered operands; a singleton is accepted outside sequence mode."""
    if isinstance(value, Sized) and not isinstance(value, (str, bytes)):
        size = len(value)
        try:
            iterator = iter(cast(Iterable[object], value))
        except TypeError as error:
            raise ValueError("Expected an iterable operand sequence") from error
        entries = tuple(islice(iterator, size + 1))
        if len(entries) != size:
            raise ValueError("Operand sequence length differs from its iteration")
        return tuple(bind_operand(entry) for entry in entries)
    if sequence_only:
        raise ValueError("Expected an operand sequence")
    return (bind_operand(value),)


def bind_operand_word_or_integer(
    value: object, width: int, *, allow_negative: bool = False
) -> tuple[_Operand, ...]:
    """Expand an integer literal, or bind a typed singleton or sequence.

    Integer offsets admit the locked signed lower bound of minus two to the
    width. Encoded negative values wrap once. Boolean scalars remain singletons.
    """
    if type(width) is not int or width < 0:
        raise ValueError("Literal width must be a nonnegative integer")
    if not isinstance(value, int) or isinstance(value, bool):
        return bind_operand_word(value)
    limit = 1 << width
    if not (-limit if allow_negative else 0) <= value < limit:
        raise ValueError("Integer literal is outside the operand word range")
    if not width:
        return ()
    return tuple(bool(bit) for bit in split_integer(int(value % limit), width))


def bind_z_word(operands: tuple[_Operand, ...]) -> tuple[_Control, ...]:
    """Require quantum wires, Z-basis slots or ordinary Boolean literals."""
    word = tuple(item for item in operands if not isinstance(item, _XBasisTarget))
    if len(word) != len(operands):
        raise ValueError("Expected a Z-basis operand word")
    return word


def bind_quantum_word(value: object, *, sequence_only: bool = False) -> tuple[int, ...]:
    """Require a qubit-only word while retaining physical order and stride."""
    operands = bind_operand_word(value, sequence_only=sequence_only)
    quantum = tuple(item for item in operands if type(item) is int)
    if len(quantum) != len(operands):
        raise ValueError("Expected a quantum-only operand word")
    return quantum
