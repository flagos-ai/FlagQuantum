# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 And and MultiAnd.on_classical_vals.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/and_bloq.py

"""Raw predicate outputs and checked erasure, independent of quantum lowering."""

import numpy as np
from numpy.typing import NDArray

from .classical_bitwise import IntegerValue


def and_classical_values(
    controls: NDArray[np.uint8],
    values: tuple[int, int] = (1, 1),
    *,
    target: IntegerValue | None = None,
    uncompute: bool = False,
) -> dict[str, IntegerValue]:
    """Create the predicate, or consume only its consistent logical output.

    Forward evaluation ignores an incoming target. Control objects are returned
    unchanged. This raw callback does not impose compiler wire or bit validation.
    """
    output = 1 if tuple(controls) == values else 0
    if not uncompute:
        return {"ctrl": controls, "target": output}
    if target != output:
        raise ValueError("Inconsistent target for erasing the And predicate")
    return {"ctrl": controls}


def multi_and_classical_values(
    controls: NDArray[np.uint8], values: tuple[int, ...]
) -> dict[str, IntegerValue]:
    """Return source prefix outputs without normalizing shape, dtype or values.

    The caller supplies concrete activation metadata. Prefix junk and target are
    logical outputs, not clean scratch. Register construction and quantum erasure
    remain separate from this forward numerical callback.
    """
    accumulated = np.bitwise_and.accumulate(
        np.equal(controls, np.asarray(values)).astype(np.uint8)
    )
    junk, target = accumulated[1:-1], accumulated[-1]
    return {"ctrl": controls, "junk": junk, "target": target}
