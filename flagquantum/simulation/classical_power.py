# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Power constructor and ordered repeated calls.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/power.py

"""Repeat a bound numerical component without replacing its execution owner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from .classical_updates import commit_phased_classical_outputs
from .value_parameters import is_symbolic_value

if TYPE_CHECKING:
    from sympy import Expr


def power_classical_values(
    values: dict[str, object],
    repetitions: int | np.integer[Any] | Expr,
    child: Callable[[dict[str, object]], tuple[dict[str, object], complex]],
    assign_outputs: Callable[[Mapping[str, object]], None],
    *,
    thru_only: bool = True,
) -> tuple[dict[str, object], complex]:
    """Retain original constructor guards and literal ordered decomposition.

    The child evaluates its complete bound numerical component, including any
    measured phase. The owner binds outputs after each call. Copies of keyword
    mappings retain their field objects. Symbolic and NumPy integer counts are
    not converted into Python counts, even if construction accepted them.
    """
    if not thru_only:
        raise ValueError("Component to repeat must have only THRU registers")
    if not is_symbolic_value(repetitions) and (
        not isinstance(repetitions, (int, np.integer)) or repetitions < 1
    ):
        raise ValueError(f"repetitions={repetitions!r} must be a positive integer")
    if not isinstance(repetitions, int):
        raise ValueError(f"Symbolic power {repetitions} not supported")
    current = dict(values)
    phase: complex = 1.0
    for _ in range(repetitions):
        outputs, contribution = child(dict(current))
        phase = commit_phased_classical_outputs(
            outputs, contribution, assign_outputs, phase=phase, outcomes={}
        )
        current = outputs
    return current, phase
