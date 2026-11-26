"""Historical validation and capability-catalog entry points."""

from collections.abc import Mapping as Mapping
from dataclasses import asdict as asdict
from dataclasses import dataclass as dataclass
from dataclasses import field as field
from importlib import import_module
from typing import TYPE_CHECKING
from typing import Any as Any

from flagquantum.runtime.planner.preflight import (
    DeploymentPreflightReport as DeploymentPreflightReport,
)
from flagquantum.runtime.planner.preflight import (
    ExecutionPreflightReport as ExecutionPreflightReport,
)
from flagquantum.runtime.planner.preflight import ValidationIssue as ValidationIssue
from flagquantum.runtime.planner.preflight import ValidationReport as ValidationReport
from flagquantum.runtime.planner.preflight import (
    preflight_deployment as preflight_deployment,
)
from flagquantum.runtime.planner.preflight import (
    preflight_execution as preflight_execution,
)

if TYPE_CHECKING:
    from flagquantum._api import (
        capabilities as capabilities,
    )
    from flagquantum.runtime.planner.preflight import (
        DeploymentPreflightReport as DeploymentPreflightReport,
    )
    from flagquantum.runtime.planner.preflight import (
        ExecutionPreflightReport as ExecutionPreflightReport,
    )
    from flagquantum.runtime.planner.preflight import (
        ValidationIssue as ValidationIssue,
    )
    from flagquantum.runtime.planner.preflight import (
        ValidationReport as ValidationReport,
    )
    from flagquantum.runtime.planner.preflight import (
        preflight_deployment as preflight_deployment,
    )
    from flagquantum.runtime.planner.preflight import (
        preflight_execution as preflight_execution,
    )

_OWNERS = {
    "ValidationIssue": "flagquantum.runtime.planner.preflight",
    "ValidationReport": "flagquantum.runtime.planner.preflight",
    "ExecutionPreflightReport": "flagquantum.runtime.planner.preflight",
    "DeploymentPreflightReport": "flagquantum.runtime.planner.preflight",
    "capabilities": "flagquantum._api",
    "preflight_execution": "flagquantum.runtime.planner.preflight",
    "preflight_deployment": "flagquantum.runtime.planner.preflight",
}

__all__ = [
    "ExecutionPreflightReport",
    "DeploymentPreflightReport",
    "ValidationIssue",
    "ValidationReport",
    "capabilities",
    "preflight_deployment",
    "preflight_execution",
]


def __getattr__(name: str) -> Any:
    if name.startswith("_"):
        raise AttributeError(name)
    if name in _OWNERS:
        value = getattr(import_module(_OWNERS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Resolve published class annotations in this public module without owner side effects.
