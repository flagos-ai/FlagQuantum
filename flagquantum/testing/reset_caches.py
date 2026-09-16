"""Reset helpers for module-level caches that would otherwise leak between tests."""

from __future__ import annotations

from ..runtime.executors.mps.reverse_planning import clear_mps_reverse_segment_cache
from ..runtime.noise_registry import reset_noise_registry
from ..simulation.mps.site_kernels import reset_site_kernel_stats
from ..simulation.real_imag_kernels import clear_kernel_caches
from ..simulation.statevector.operations import clear_statevector_layout_cache

__all__ = ("reset_caches",)


def reset_caches() -> None:
    """Clear module-level mutable caches so tests cannot observe cross-test state."""

    reset_site_kernel_stats(clear_cache=True)
    clear_mps_reverse_segment_cache()
    clear_kernel_caches()
    clear_statevector_layout_cache()
    reset_noise_registry()
