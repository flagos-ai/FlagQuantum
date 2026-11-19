# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _infra/composite_bloq.py _map_soqs.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/composite_bloq.py

"""Map scalar and shaped register handles without consuming linear values."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import numpy as np

from ...core.ir.program.register_bindings import RegisterBinding

if TYPE_CHECKING:
    from .component_builder import BindingCollection


def map_register_bindings(
    bindings: Mapping[str, BindingCollection],
    replacements: Iterable[tuple[BindingCollection, BindingCollection]],
) -> dict[str, BindingCollection]:
    """Apply ordered replacement pairs, keeping unmapped handles unchanged.

    Later duplicate pairs replace earlier ones. Both sides of each pair must
    have the same scalar or array form, and arrays must have equal shapes.
    Mapping creates a fresh result array and never consumes the input handles.
    """
    flat: dict[RegisterBinding, RegisterBinding] = {}
    for old, new in replacements:
        if isinstance(old, RegisterBinding):
            assert isinstance(new, RegisterBinding), new
            flat[old] = new
        else:
            assert isinstance(old, np.ndarray), old
            assert isinstance(new, np.ndarray), new
            assert old.shape == new.shape, (old.shape, new.shape)
            flat.update(zip(old.flat, new.flat, strict=True))

    def mapped(value: RegisterBinding) -> RegisterBinding:
        return flat.get(value, value)

    vectorized = np.vectorize(mapped, otypes=[object])
    return {
        name: mapped(value) if isinstance(value, RegisterBinding) else vectorized(value)
        for name, value in bindings.items()
    }
