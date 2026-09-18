"""Reusable quantum primitives shared by the algorithm modules.

Primitives build :class:`~flagquantum.circuit.Circuit` objects from other circuits and from
classical data. They do not define circuit or operator semantics, select a runtime, or
introduce an execution path of their own.

A primitive is admitted here only when at least two algorithm modules need it; this package
is not a general-purpose quantum toolkit.
"""

from __future__ import annotations

from .qft import append_qft as append_qft
from .qft import qft as qft

__all__ = ["append_qft", "qft"]
