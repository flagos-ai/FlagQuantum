# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QROMBase.on_classical_vals.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/data_loading/qrom_base.py

"""Classical register-value evaluation for XOR lookup and checked erasure."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from operator import xor
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ..core.value_math import SymbolicInt
from .classical_bitwise import IntegerValue

if TYPE_CHECKING:
    from sympy import Basic

LookupValue: TypeAlias = "IntegerValue | Basic"


def lookup_classical_values(
    data: tuple[NDArray[np.generic], ...] | None,
    selection_shape: tuple[int, ...],
    values: Mapping[str, LookupValue],
    *,
    num_controls: int = 0,
    erase: bool = False,
) -> dict[str, LookupValue]:
    """Evaluate the locked lookup's named classical values without quantum gates.

    Dataset axes concatenate selection and target axes. The control word must
    be all ones to activate the lookup. Erasure verifies zero after XOR and
    removes target outputs. The caller retains its arrays and input mapping.

    The original omitted-singleton indexing and scalar-array failures are
    retained. This classical model is not a proof of quantum semantics on
    those source failure boundaries.
    """
    if data is None:
        raise NotImplementedError(
            "Unknown-data lookup cannot evaluate classical values"
        )
    if (
        not data
        or type(num_controls) is not int
        or num_controls < 0
        or type(erase) is not bool
        or any(type(n) is not int or n < 1 for n in selection_shape)
    ):
        raise ValueError(
            "Classical lookup requires data, positive axes and control width"
        )
    vals = dict(values)
    controls: dict[str, LookupValue] = {}
    if num_controls > 0:
        control = vals["control"]
        if control != (1 << num_controls) - 1:
            return vals
        controls = {"control": control}

    return controls | _lookup_values(
        data, sum(n > 1 for n in selection_shape), vals, erase=erase
    )


def source_lookup_classical_values(
    data: tuple[NDArray[np.generic], ...] | None,
    selection_axes: Callable[[], tuple[SymbolicInt, ...]],
    values: Mapping[str, LookupValue],
    *,
    num_controls: SymbolicInt = 0,
    erase: bool = False,
) -> dict[str, LookupValue]:
    """Evaluate source metadata, reading deferred axes only after activation.

    Negative control metadata follows the source's uncontrolled callback. Boolean
    and symbolic control metadata retain its comparison and bypass behavior.
    This host-value protocol does not authorize physical negative-width storage.
    """
    if data is None:
        raise NotImplementedError(
            "Unknown-data lookup cannot evaluate classical values"
        )
    if type(erase) is not bool:
        raise ValueError("Classical lookup erasure must be boolean")
    vals = dict(values)
    controls: dict[str, LookupValue] = {}
    if num_controls > 0:
        control = vals["control"]
        if control != 2**num_controls - 1:
            return vals
        controls = {"control": control}
    return controls | _lookup_values(data, len(selection_axes()), vals, erase=erase)


def _lookup_values(
    data: tuple[NDArray[np.generic], ...],
    dimensions: int,
    vals: dict[str, LookupValue],
    *,
    erase: bool,
) -> dict[str, LookupValue]:
    selections: dict[str, LookupValue]
    index: int | tuple[int, ...]
    if dimensions == 0:
        index, selections = 0, {}
    elif dimensions == 1:
        index = int(cast(int, vals.pop("selection", 0)))
        selections = {"selection": index}
    else:
        index = tuple(int(cast(int, vals[f"selection{i}"])) for i in range(dimensions))
        selections = {f"selection{i}": index[i] for i in range(dimensions)}

    targets: dict[str, LookupValue] = {
        f"target{i}_": array[index] for i, array in enumerate(data)
    }
    targets = {
        name: cast(LookupValue, xor(value, vals.get(name, 0)))
        for name, value in targets.items()
    }
    if erase:
        for name, value in targets.items():
            if np.any(value):
                raise ValueError(
                    f"Target register {name} must be uncomputed before de-allocation. "
                    f"Found values {value}"
                )
        targets = {}
    return selections | targets
