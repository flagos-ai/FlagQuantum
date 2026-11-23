# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix py/circuit/ops/op_testing_util.py.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Share complete operand families and packed truth-table patterns in tests."""

from collections.abc import Sized
from typing import cast

from ..compiler.artifacts.operand_identity import OperandIdentity, OperandKind
from ..compiler.registers.layout.operand_word import bind_operand_array
from ..ecosystem._physical_basis_session import PhysicalBasisSession

U64_FALSE = 0
U64_TRUE = (1 << 64) - 1
U64_BIT5 = 0xFFFFFFFF00000000
U64_BIT4 = 0xFFFF0000FFFF0000
U64_BIT3 = 0xFF00FF00FF00FF00
U64_BIT2 = 0xF0F0F0F0F0F0F0F0
U64_BIT1 = 0xCCCCCCCCCCCCCCCC
U64_BIT0 = 0xAAAAAAAAAAAAAAAA


def store64(
    sim: PhysicalBasisSession, index: OperandIdentity | bool, val64: int
) -> int:
    """Store a lane mask, retaining literal constants and X-basis slot meaning."""
    if isinstance(index, bool):
        return U64_TRUE if index else U64_FALSE
    if index.kind in (OperandKind.BOOLEAN, OperandKind.X_BOOLEAN):
        return U64_TRUE if bool(index) else U64_FALSE
    if index.kind is OperandKind.X_CLASSICAL:
        index = OperandIdentity(OperandKind.CLASSICAL, index.value)
    sim.write_across_shots(index, val64)
    return val64


def read64(sim: PhysicalBasisSession, index: OperandIdentity | bool) -> int:
    """Read X-basis classical operands through their underlying bit storage."""
    if isinstance(index, bool):
        return U64_TRUE if index else U64_FALSE
    if index.kind in (OperandKind.BOOLEAN, OperandKind.X_BOOLEAN):
        return U64_TRUE if bool(index) else U64_FALSE
    if index.kind is OperandKind.X_CLASSICAL:
        index = OperandIdentity(OperandKind.CLASSICAL, index.value)
    value = sim.read_across_shots(index, out=int)
    assert isinstance(value, int)
    return value


def x_cases_q(index: int) -> list[OperandIdentity]:
    return [OperandIdentity(OperandKind.QUANTUM, index)]


def x_cases_bc(index: int) -> list[OperandIdentity]:
    return [
        OperandIdentity(OperandKind.X_CLASSICAL, index),
        OperandIdentity(OperandKind.X_BOOLEAN, 0),
        OperandIdentity(OperandKind.X_BOOLEAN, 1),
    ]


def z_cases(index: int) -> list[OperandIdentity | bool]:
    return [
        OperandIdentity(OperandKind.QUANTUM, index),
        OperandIdentity(OperandKind.CLASSICAL, index),
        False,
        True,
    ]


def x_list_cases(index: int, size: int, *, include_x_bits: bool = True) -> list[object]:
    positions = range(index, index + size)
    result: list[object] = [
        [OperandIdentity(OperandKind.QUANTUM, k) for k in positions]
    ]
    result.append(bind_operand_array(result[-1]))
    if include_x_bits:
        result.append(
            bind_operand_array(
                [OperandIdentity(OperandKind.X_CLASSICAL, k) for k in positions]
            )
        )
        result.append(
            bind_operand_array(
                [
                    OperandIdentity(
                        OperandKind.X_BOOLEAN, int(hash((k, index, size)) & 1 == 0)
                    )
                    for k in positions
                ]
            )
        )
    mixed = []
    for k in positions:
        choice = hash((k, index, size)) % 4
        if choice == 0 and include_x_bits:
            mixed.append(OperandIdentity(OperandKind.X_BOOLEAN, 0))
        elif choice == 1 and include_x_bits:
            mixed.append(OperandIdentity(OperandKind.X_BOOLEAN, 1))
        elif choice == 2 and include_x_bits:
            mixed.append(OperandIdentity(OperandKind.X_CLASSICAL, k))
        else:
            mixed.append(OperandIdentity(OperandKind.QUANTUM, k))
    result.append(bind_operand_array(mixed))
    result.extend(x_cases_q(index))
    if include_x_bits:
        result.extend(x_cases_bc(index))
    return result


def z_list_cases(index: int, size: int) -> list[object]:
    positions = range(index, index + size)
    result: list[object] = [
        [OperandIdentity(OperandKind.QUANTUM, k) for k in positions]
    ]
    result.append(bind_operand_array(result[-1]))
    result.append(
        bind_operand_array(
            [OperandIdentity(OperandKind.CLASSICAL, k) for k in positions]
        )
    )
    result.append(
        bind_operand_array([hash((k, index, size)) & 1 == 0 for k in positions])
    )
    mixed: list[OperandIdentity | bool] = []
    for k in positions:
        choice = hash((k, index, size)) % 4
        if choice == 0:
            mixed.append(False)
        elif choice == 1:
            mixed.append(True)
        elif choice == 2:
            mixed.append(OperandIdentity(OperandKind.QUANTUM, k))
        else:
            mixed.append(OperandIdentity(OperandKind.CLASSICAL, k))
    result.append(bind_operand_array(mixed))
    result.extend(z_cases(index))
    return result


def broadcast_logic(*xs: object) -> list[object]:
    for value in xs:
        if isinstance(value, Sized):
            count = len(value)
            break
    else:
        count = 1
    result = [
        (
            [value] * count
            if isinstance(value, bool)
            or isinstance(value, OperandIdentity)
            and value.kind is not OperandKind.REGISTER
            else value
        )
        for value in xs
    ]
    for value in result:
        assert len(cast(Sized, value)) == count
    return result
