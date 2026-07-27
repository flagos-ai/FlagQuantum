"""Deterministic kernel selection policy for the statevector backend."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

from .environment import mode


@dataclass(frozen=True)
class KernelDecision:
    """One auditable accelerated-kernel selection decision."""

    feature: str
    selected: str
    reason: str

    @property
    def accelerated(self) -> bool:
        return self.selected == "triton"

    def summary(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "selected": self.selected,
            "accelerated": self.accelerated,
            "reason": self.reason,
        }


def triton_available() -> bool:
    """Return whether the optional Triton runtime is importable."""

    return importlib.util.find_spec("triton") is not None


def select_triton_kernel(
    feature: str,
    *,
    requested: bool,
    supported: bool = True,
    available: bool | None = None,
) -> KernelDecision:
    """Select Triton or the portable PyTorch implementation with a reason."""

    if mode() == "portable":
        return KernelDecision(feature, "pytorch", "portable_mode")
    if not requested:
        return KernelDecision(feature, "pytorch", "disabled_by_policy")
    resolved_available = triton_available() if available is None else bool(available)
    if not resolved_available:
        return KernelDecision(feature, "pytorch", "triton_unavailable")
    if not supported:
        return KernelDecision(feature, "pytorch", "input_not_supported")
    return KernelDecision(feature, "triton", "eligible")


__all__ = ("KernelDecision", "select_triton_kernel", "triton_available")
