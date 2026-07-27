"""JAX-specific execution collectors, separate from evidence schemas."""

from __future__ import annotations

from typing import Any

_EXPORTS = {
    "collect_mps_accelerator_backward_evidence": "_collect_jax_mps_accelerator_backward_evidence",
    "execute_minimal_mps_sharded_backward": "_execute_minimal_mps_sharded_backward",
    "execute_minimal_mps_sharded_optimizer_step": "_execute_minimal_mps_sharded_optimizer_step",
}


def __getattr__(name: str) -> Any:
    legacy_name = _EXPORTS.get(name)
    if legacy_name is None:
        raise AttributeError(name)
    from ._loader import resolve

    return resolve(legacy_name, domain="mps")


def __dir__() -> list[str]:
    return sorted(_EXPORTS)


__all__ = sorted(_EXPORTS)
