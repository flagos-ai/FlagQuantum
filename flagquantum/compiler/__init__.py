"""Stable expert compiler interface.

Normal execution compiles internally through :func:`flagquantum.plan`.
"""

from .noise import channel_instruction, lower_noise_model
from .pipeline import (
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
    "channel_instruction",
    "lower_noise_model",
    "merge_adjacent_rotations",
    "merge_self_inverse",
    "remove_identity_gates",
    "route_to_topology",
    "schedule_layers",
    "simple_compile",
)
