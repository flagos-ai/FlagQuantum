"""Stable expert compiler interface.

Normal execution compiles internally through :func:`flagquantum.plan`.
"""

from .compilation.compiler import (
    CouplingMap,
    compile_for_backend,
    merge_adjacent_rotations,
    merge_self_inverse,
    remove_identity_gates,
    route_to_topology,
    schedule_layers,
    simple_compile,
)

compile = compile_for_backend

__all__ = (
    "CouplingMap",
    "compile",
    "compile_for_backend",
    "merge_adjacent_rotations",
    "merge_self_inverse",
    "remove_identity_gates",
    "route_to_topology",
    "schedule_layers",
    "simple_compile",
)
