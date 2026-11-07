# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _cvs_convert.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/controlled.py

"""Group raw activation values before bound classical control evaluation."""

from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np

from .value_dimensions import ArrayValue, SymbolicArrayShape

ControlPatternInput: TypeAlias = (
    int | np.integer[Any] | ArrayValue | Sequence["ControlPatternInput"]
)


def normalize_control_patterns(values: ControlPatternInput) -> tuple[ArrayValue, ...]:
    """Preserve source grouping, array identity and symbolic shape descriptors.

    An integer sequence is one array register; nested sequences are separate
    register clauses. This converter does not validate activation domains,
    widths or register counts. Those belong to the bound consumer.
    """
    if isinstance(values, SymbolicArrayShape):
        return (values,)
    if isinstance(values, (int, np.integer)):
        return (np.array(values),)
    if isinstance(values, np.ndarray):
        return (values,)
    if all(isinstance(value, (int, np.integer)) for value in values):
        return (np.asarray(values),)
    return tuple(
        value if isinstance(value, SymbolicArrayShape) else np.asarray(value)
        for value in values
    )
