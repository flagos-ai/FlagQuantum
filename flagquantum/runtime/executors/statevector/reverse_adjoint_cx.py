"""CPU CX-segment helpers for the reversible adjoint sweep."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from flagquantum.simulation.statevector.cx_adjoint import (
    apply_cpu_compact_cx_adjoint_segment as apply_cpu_compact_cx_adjoint_segment,
)
from flagquantum.simulation.statevector.cx_adjoint import (
    apply_cpu_cx_adjoint_index as apply_cpu_cx_adjoint_index,
)
from flagquantum.simulation.statevector.cx_adjoint import (
    apply_cpu_cx_adjoint_segment as apply_cpu_cx_adjoint_segment,
)
from flagquantum.simulation.statevector.cx_adjoint import (
    cpu_cx_permutation_images as cpu_cx_permutation_images,
)
from flagquantum.simulation.statevector.cx_adjoint import (
    cpu_cx_permutation_index as cpu_cx_permutation_index,
)

from ....simulation.native_cpu import (
    compact_cpu_cx_adjoint_auxiliary_bytes,
    fused_compact_cx_adjoint_inplace_,
    fused_cx_adjoint_inplace_,
    native_cpu_cx_rotation_adjoint_fusion_available,
    use_compact_cpu_cx_adjoint_cycles,
    use_compact_cpu_cx_mapping,
)


def apply_or_defer_cpu_cx_adjoint_segment(
    sweep: Any,
    segment_qubits: Sequence[Sequence[int]],
    *,
    segment_start: int,
    index: int,
) -> bool:
    """Apply a CPU CX segment or defer it into the preceding rotation layer."""

    controls = tuple(qubits[0] for qubits in reversed(segment_qubits))
    targets = tuple(qubits[1] for qubits in reversed(segment_qubits))
    if sweep.policy.low_memory_cpu_cx:
        images = cpu_cx_permutation_images(
            controls,
            targets,
            sweep.plan.n_qubits,
        )
        if (
            sweep.policy.compact_cpu_cx_cycles
            and use_compact_cpu_cx_adjoint_cycles(len(controls), sweep.plan.n_qubits)
            and fused_compact_cx_adjoint_inplace_(
                sweep.reversible_state.amplitudes,
                sweep.adjoint,
                images,
            )
        ):
            sweep.evidence.peak_scratch_bytes = max(
                sweep.evidence.peak_scratch_bytes,
                compact_cpu_cx_adjoint_auxiliary_bytes(sweep.plan.n_qubits),
            )
            sweep.skipped_cx_indices.update(range(segment_start, index))
            return True
        if fused_cx_adjoint_inplace_(
            sweep.reversible_state.amplitudes,
            sweep.adjoint,
            controls,
            targets,
            sweep.plan.n_qubits,
        ):
            sweep.skipped_cx_indices.update(range(segment_start, index))
            return True
    if (
        segment_start > 0
        and sweep.bound.instructions[segment_start - 1].name in {"rx", "ry", "rz"}
        and sweep.pending_cpu_cx_index is None
        and sweep.pending_cpu_cx_images is None
        and native_cpu_cx_rotation_adjoint_fusion_available()
    ):
        if use_compact_cpu_cx_mapping(sweep.plan.n_qubits):
            sweep.pending_cpu_cx_images = cpu_cx_permutation_images(
                controls,
                targets,
                sweep.plan.n_qubits,
            )
            sweep.pending_cpu_cx_controls = controls
            sweep.pending_cpu_cx_targets = targets
        else:
            sweep.pending_cpu_cx_index = cpu_cx_permutation_index(
                sweep.reversible_state.amplitudes,
                controls,
                targets,
                sweep.plan.n_qubits,
            )
    else:
        ket, adjoint = apply_cpu_cx_adjoint_segment(
            sweep.reversible_state.amplitudes,
            sweep.adjoint,
            controls,
            targets,
            sweep.plan.n_qubits,
        )
        sweep.reversible_state = replace(sweep.reversible_state, amplitudes=ket)
        sweep.adjoint = adjoint
        sweep.evidence.peak_scratch_bytes = max(
            sweep.evidence.peak_scratch_bytes,
            2 * ket.numel() * ket.element_size(),
        )
    sweep.skipped_cx_indices.update(range(segment_start, index))
    return True
