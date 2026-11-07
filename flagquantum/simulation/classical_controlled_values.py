# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _ControlledBase value and basis-phase callbacks.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/controlled.py

"""Delegate bound controlled values and phases without a foreign graph hierarchy."""

from collections.abc import Mapping, Sequence
from types import NotImplementedType
from typing import TypeVar

from .classical_action import ClassicalValueCallback
from .classical_control_activation import classical_controls_active
from .value_dimensions import ArrayValue

Result = TypeVar("Result")


def _controlled_inputs(
    values: Mapping[str, object],
    control_names: Sequence[str],
    child_names: Sequence[str],
) -> tuple[list[object], dict[str, object]]:
    controls = [values[name] for name in control_names]
    child_values = {name: values[name] for name in child_names}
    return controls, child_values


def controlled_classical_values(
    values: Mapping[str, object],
    control_names: Sequence[str],
    child_names: Sequence[str],
    patterns: Sequence[ArrayValue],
    child: ClassicalValueCallback[dict[str, object] | NotImplementedType],
    *,
    thru_only: bool = True,
    symbolic_parameters: Sequence[object] = (),
) -> dict[str, object] | NotImplementedType:
    """Apply the source THRU guard, field extraction and conditional delegation.

    Copy the named mapping as the original **kwargs boundary does. Active child
    dictionaries are updated in place with original control objects. Inactive
    calls return the copied values including extras; NotImplemented is retained.
    """
    named = dict(values)
    if not thru_only:
        raise ValueError("Controlled numerical evaluation requires THRU registers")
    controls, child_values = _controlled_inputs(named, control_names, child_names)
    if classical_controls_active(
        controls, patterns, symbolic_parameters=symbolic_parameters
    ):
        outputs = child(**child_values)
        if outputs is not NotImplemented:
            outputs |= dict(zip(control_names, controls, strict=False))
        return outputs
    return named


def controlled_basis_phase(
    values: Mapping[str, object],
    control_names: Sequence[str],
    child_names: Sequence[str],
    patterns: Sequence[ArrayValue],
    child: ClassicalValueCallback[Result],
    *,
    symbolic_parameters: Sequence[object] = (),
) -> Result | None:
    """Extract both sets of fields before activation; pass active phases unchanged.

    The original phase callback has no THRU-only guard and returns None when
    inactive. This function does not reinterpret None or evaluate opaque child
    measurement-phase descriptors.
    """
    controls, child_values = _controlled_inputs(values, control_names, child_names)
    if classical_controls_active(
        controls, patterns, symbolic_parameters=symbolic_parameters
    ):
        return child(**child_values)
    return None
