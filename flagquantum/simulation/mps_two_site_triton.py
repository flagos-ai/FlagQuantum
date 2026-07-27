"""Compatibility alias for :mod:`.triton_kernels.mps_two_site`."""

from __future__ import annotations

import sys

from .triton_kernels import mps_two_site as _implementation

sys.modules[__name__] = _implementation
