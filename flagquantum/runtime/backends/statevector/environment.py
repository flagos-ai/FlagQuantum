"""Statevector runtime environment configuration with compatibility aliases."""

from __future__ import annotations

import os

_ALIASES = {
    "FQ_SV_PERSISTENT_LAYOUT": "FQ_STATEVECTOR_PERSISTENT_WIRE_LAYOUT",
    "FQ_SV_TRITON_LOCAL_CX": "FQ_STATEVECTOR_TRITON_LOCAL_CX",
    "FQ_SV_TRITON_CX_SEGMENT": "FQ_STATEVECTOR_TRITON_CX_SEGMENT",
    "FQ_SV_LOCAL_FUSION": "FQ_STATEVECTOR_LOCAL_BLOCK_FUSION",
    "FQ_SV_LOCAL_FUSION_WIDTH": "FQ_STATEVECTOR_LOCAL_BLOCK_FUSION_WIDTH",
    "FQ_SV_INTER_NODE_CHECKPOINTS": "FQ_STATEVECTOR_INTER_NODE_KET_CHECKPOINTS",
    "FQ_SV_CROSS_SHARD_CX_PACK": "FQ_STATEVECTOR_CROSS_SHARD_CX_PACK",
}


def get(name: str, default: str) -> str:
    """Read canonical name first, then its pre-1.0 compatibility alias."""

    value = os.getenv(name)
    if value is not None:
        return value
    legacy = _ALIASES.get(name)
    return os.getenv(legacy, default) if legacy else default


def get_bool(name: str, default: bool) -> bool:
    return get(name, "1" if default else "0").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def mode() -> str:
    value = os.getenv("FQ_SV_RUNTIME_MODE", "auto").strip().lower()
    if value not in {"auto", "portable", "max_perf"}:
        raise ValueError("FQ_SV_RUNTIME_MODE must be auto, portable, or max_perf")
    return value


__all__ = ("get", "get_bool", "mode")
