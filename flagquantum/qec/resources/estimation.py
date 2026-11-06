"""Report analytical fault-tolerant resources alongside optional emitted IR."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ...compiler._component_trace.resource_comparison import (
    EmittedInstructionCounts,
    count_emitted_instructions,
)
from ...compiler._component_trace.resource_model import (
    ComponentCost,
    ModeledResources,
    estimate_component_resources,
)
from ...core.ir import CircuitIR
from .beverland import BeverlandEstimate, estimate_beverland_resources


@runtime_checkable
class _HasConcreteResourceModel(Protocol):
    def resource_model(self) -> ComponentCost: ...


@dataclass(frozen=True)
class FTQCResourceReport:
    """Keep the declared logical model, emitted IR, and physical scenario separate."""

    component: str
    logical: ModeledResources
    emitted: EmittedInstructionCounts | None
    surface_code: BeverlandEstimate | None


def estimate_ftqc_resources(
    component: ComponentCost | _HasConcreteResourceModel,
    *,
    emitted_ir: CircuitIR | None = None,
    error_budget: float | None = None,
    physical_error: float | None = None,
) -> FTQCResourceReport:
    """Estimate a concrete component model or an operation that provides one.

    An operation's zero-argument `resource_model()` supplies its default
    model. Pass a concrete `ComponentCost` to select a different direction or
    provide costs for composed operations with explicit model inputs.
    Supply an already emitted circuit for a side-by-side instruction count.
    The caller pairs that circuit with the component model. Supply both error
    rates to estimate a Beverland fast data block from the logical model.
    """
    if (error_budget is None) != (physical_error is None):
        raise ValueError("Supply both error_budget and physical_error")
    if isinstance(component, _HasConcreteResourceModel):
        component = component.resource_model()
    if not isinstance(component, ComponentCost):
        raise TypeError("FTQC estimation requires a concrete ComponentCost model")
    logical = estimate_component_resources(component)
    surface_code = None
    if error_budget is not None and physical_error is not None:
        surface_code = estimate_beverland_resources(
            logical, error_budget=error_budget, physical_error=physical_error
        )
    emitted = None
    if emitted_ir is not None:
        emitted = count_emitted_instructions(emitted_ir.validate())
    return FTQCResourceReport(component.name, logical, emitted, surface_code)
