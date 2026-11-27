"""Optional optimized kernels used by FlagQuantum runtimes.

Importing this package does not import an optional kernel compiler. Runtime
policy selects a concrete implementation before importing its launch wrapper.
"""

from __future__ import annotations

__all__: list[str] = []
