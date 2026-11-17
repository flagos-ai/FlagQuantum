# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 rotation_synthesis/_typing.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/rotation_synthesis/_typing.py

"""Internal scalar types used by rotation synthesis."""

from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from typing_extensions import TypeIs

if TYPE_CHECKING:
    from mpmath.ctx_mp_python import mpc, mpf

Real: TypeAlias = "float | np.floating[Any] | mpf"
Integral: TypeAlias = int | np.integer[Any]
Complex: TypeAlias = "complex | np.complexfloating[Any, Any] | mpc"


def is_int(value: object) -> TypeIs[Integral]:
    return isinstance(value, (int, np.integer))
