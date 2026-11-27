"""Phase-aware no-progress diagnosis independent of process launchers."""

from __future__ import annotations

from dataclasses import dataclass as dataclass

from flagquantum.runtime.observability.watchdog import (
    PhaseAwareWatchdog as PhaseAwareWatchdog,
)
from flagquantum.runtime.observability.watchdog import (
    ProgressSnapshot as ProgressSnapshot,
)
from flagquantum.runtime.observability.watchdog import (
    StallDiagnosis as StallDiagnosis,
)
