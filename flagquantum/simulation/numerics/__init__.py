"""Experimental simulation numerical building blocks.

These primitives do not select a runtime or claim that float32 hardware can
execute a FlagQuantum complex128 workload. Runtime adoption requires a
separate precision provider and accuracy certification.
"""

from .conformance import (
    DoubleSingleConformanceCase,
    DoubleSingleConformanceReport,
    run_double_single_conformance,
)
from .double_single import (
    DoubleSingleComplexTensor,
    DoubleSingleTensor,
    double_single_dot,
    double_single_sum,
    quick_two_sum,
    split_float32,
    two_prod,
    two_sum,
)

__all__ = (
    "DoubleSingleComplexTensor",
    "DoubleSingleTensor",
    "DoubleSingleConformanceCase",
    "DoubleSingleConformanceReport",
    "double_single_dot",
    "double_single_sum",
    "quick_two_sum",
    "split_float32",
    "two_prod",
    "two_sum",
    "run_double_single_conformance",
)
