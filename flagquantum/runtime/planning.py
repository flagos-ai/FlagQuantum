"""Runtime selection and hybrid-parallel planning contracts."""

from .parallel import (
    HybridParallelPlan,
    ObservableGroup,
    group_observables,
    plan_hybrid_parallel,
)

__all__ = (
    "HybridParallelPlan",
    "ObservableGroup",
    "group_observables",
    "plan_hybrid_parallel",
)
