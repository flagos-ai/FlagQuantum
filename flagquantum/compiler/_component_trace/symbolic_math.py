# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 symbolics.math_funcs.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/symbolics/math_funcs.py

"""Resource-facing imports of shared scalar and symbolic width mathematics."""

from ...core.value_math import SymbolicInt as SymbolicInt
from ...core.value_math import SymbolicReal as SymbolicReal
from ...core.value_math import bit_length as bit_length
from ...core.value_math import ceil as ceil
from ...core.value_math import floor as floor
from ...core.value_math import log2 as log2
from ...core.value_math import smax as smax
from ...core.value_math import smin as smin
