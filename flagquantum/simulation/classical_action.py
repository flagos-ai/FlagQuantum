# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ClassicalSimState.step callback dispatch.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/classical_sim.py

"""Dispatch bound numerical actions before graph-owned output assignment."""

from collections.abc import Callable
from typing import Protocol, TypeVar

Value = TypeVar("Value", covariant=True)
Phase = TypeVar("Phase")


class ClassicalValueCallback(Protocol[Value]):
    """A named numerical callback shared by direct and controlled actions."""

    def __call__(self, **values: object) -> Value: ...


def evaluate_classical_action(
    values: dict[str, object],
    value_action: ClassicalValueCallback[object],
    phase_action: ClassicalValueCallback[Phase],
    evaluate_decomposition: Callable[
        [dict[str, object]], tuple[dict[str, object], Phase]
    ],
    *,
    operation_name: str,
) -> tuple[dict[str, object], Phase]:
    """Call values, then phase; recurse only for NotImplemented and no phase.

    Direct output dictionaries and phase objects retain their identity. The
    decomposition evaluator receives the original input mapping. Graph traversal,
    decomposition-exception translation, output binding and phase application
    remain with their existing owners; this helper performs none of them.
    """
    outputs = value_action(**values)
    phase = phase_action(**values)
    if outputs is NotImplemented:
        if phase is not None:
            raise ValueError(
                f"`basis_state_phase` defined on {operation_name}, "
                "but not `on_classical_vals`"
            )
        return evaluate_decomposition(values)
    if not isinstance(outputs, dict):
        raise TypeError(
            f"`{operation_name}.on_classical_vals` should return a dictionary."
        )
    return outputs, phase
