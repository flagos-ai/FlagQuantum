"""Stable expert compiler interface.

Normal execution compiles internally through :func:`flagquantum.plan`.
"""

from .noise import channel_instruction, lower_noise_model
from .pipeline import (
    CouplingMap,
    compile,
    merge_adjacent_rotations,
    merge_self_inverse,
    optimize,
    remove_identity_gates,
    route_to_topology,
    schedule_layers,
)

__all__ = (
    "CouplingMap",
    "compile",
    "channel_instruction",
    "lower_noise_model",
    "merge_adjacent_rotations",
    "merge_self_inverse",
    "optimize",
    "remove_identity_gates",
    "route_to_topology",
    "schedule_layers",
)
