"""Small adapters used to normalize backend-native execution metadata."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Mapping


class LiveRuntimeSummary(Mapping[str, Any]):
    """Expose mutable native evidence through a stable Mapping contract."""

    def __init__(self, source: Any, overrides: Mapping[str, Any] | None = None) -> None:
        self._source = source
        self._overrides = dict(overrides or {})

    def _snapshot(self) -> dict[str, Any]:
        return {**dict(self._source.summary()), **self._overrides}

    def __getitem__(self, key: str) -> Any:
        return self._snapshot()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._snapshot())

    def __len__(self) -> int:
        return len(self._snapshot())


__all__ = ("LiveRuntimeSummary",)
