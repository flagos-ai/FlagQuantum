# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 XGate, ZGate and CZ value/phase methods.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/x_basis.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/z_basis.py

"""Classical bit actions and separate diagonal basis-state phases."""

from typing import TypeVar

from .classical_bitwise import IntegerValue

Value = TypeVar("Value")


def x_classical_value(value: IntegerValue) -> IntegerValue:
    """Use the source addition and modulo order without binary validation."""
    return (value + 1) % 2


def diagonal_classical_values(**values: Value) -> dict[str, Value]:
    """Return the original named values, including extras, without phase loss claims."""
    return values


def z_basis_phase(value: IntegerValue) -> int:
    """Return the source sign using exact equality with one, not truthiness."""
    if value == 1:
        return -1
    return 1


def cz_basis_phase(control: IntegerValue, target: IntegerValue) -> int:
    """Retain the original left-to-right equality-test short circuit."""
    if control == 1 and target == 1:
        return -1
    return 1
