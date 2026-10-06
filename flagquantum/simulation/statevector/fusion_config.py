"""Private CPU statevector fusion configuration and measured widths."""

from __future__ import annotations

import torch

from .operations import _CPU_DISJOINT_DENSE_MAX_WIRES, _environment_flag

_CPU_DISJOINT_DENSE_MEDIUM_STATE_MIN_WIRES = 18
_CPU_DISJOINT_DENSE_MEDIUM_STATE_MAX_WIRES = 6
_CPU_DISJOINT_DENSE_FOUR_WIRE_CROSSOVER = 20
_CPU_DISJOINT_DENSE_EIGHT_WIRE_STATE_WIRES = 22
_CPU_DISJOINT_DENSE_LARGE_STATE_MAX_WIRES = 8


def _cpu_controlled_phase_decomposition_fusion_enabled() -> bool:
    return _environment_flag(
        "FQ_CPU_CONTROLLED_PHASE_DECOMPOSITION_FUSION", default=True
    )


def _cpu_controlled_phase_graph_fusion_enabled() -> bool:
    return _environment_flag("FQ_CPU_CONTROLLED_PHASE_GRAPH_FUSION", default=True)


def _cpu_hadamard_controlled_phase_fusion_enabled() -> bool:
    return _environment_flag("FQ_CPU_HADAMARD_CONTROLLED_PHASE_FUSION", default=True)


def _cpu_adaptive_dense_fusion_width_enabled() -> bool:
    return _environment_flag("FQ_CPU_ADAPTIVE_DENSE_FUSION_WIDTH", default=True)


def _cpu_cz_graph_fusion_enabled() -> bool:
    return _environment_flag("FQ_CPU_CZ_GRAPH_FUSION", default=True)


def _cpu_disjoint_dense_max_wires(
    n_wires: int,
    batch_size: int,
    dtype: torch.dtype,
) -> int:
    """Choose the measured dense-fusion width for this CPU state shape."""

    adaptive = _cpu_adaptive_dense_fusion_width_enabled()
    batched_medium = batch_size > 1 and n_wires in {18, 19}
    if dtype == torch.complex128 and adaptive and batched_medium:
        return _CPU_DISJOINT_DENSE_MEDIUM_STATE_MAX_WIRES
    if batch_size == 1 and dtype == torch.complex128:
        if not adaptive:
            if n_wires >= _CPU_DISJOINT_DENSE_EIGHT_WIRE_STATE_WIRES:
                return _CPU_DISJOINT_DENSE_MEDIUM_STATE_MAX_WIRES
            return _CPU_DISJOINT_DENSE_MAX_WIRES
        if n_wires == _CPU_DISJOINT_DENSE_EIGHT_WIRE_STATE_WIRES:
            return _CPU_DISJOINT_DENSE_LARGE_STATE_MAX_WIRES
        if (
            n_wires >= _CPU_DISJOINT_DENSE_MEDIUM_STATE_MIN_WIRES
            and n_wires != _CPU_DISJOINT_DENSE_FOUR_WIRE_CROSSOVER
        ):
            return _CPU_DISJOINT_DENSE_MEDIUM_STATE_MAX_WIRES
    return _CPU_DISJOINT_DENSE_MAX_WIRES
