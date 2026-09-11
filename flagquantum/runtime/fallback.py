"""Explicit fallback policy for heterogeneous platform execution."""

from __future__ import annotations

from enum import Enum


class FallbackPolicy(str, Enum):
    """Controls whether execution may leave the requested device."""

    FORBID = "forbid"
    SAME_DEVICE_PORTABLE = "same_device_portable"
    HOST_DEBUG_ONLY = "host_debug_only"

    @classmethod
    def normalize(cls, value: "FallbackPolicy | str") -> "FallbackPolicy":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"unsupported fallback policy {value!r}; choose one of: {choices}"
            ) from exc


__all__ = ["FallbackPolicy"]
