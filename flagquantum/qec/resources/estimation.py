"""Report analytical fault-tolerant resources alongside optional emitted IR."""

from dataclasses import dataclass

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


@dataclass(frozen=True)
class FTQCResourceReport:
    """Keep the declared logical model, emitted IR, and physical scenario separate."""

    component: str
    logical: ModeledResources
    emitted: EmittedInstructionCounts | None
    surface_code: BeverlandEstimate | None


def estimate_ftqc_resources(
    component: ComponentCost,
    *,
    emitted_ir: CircuitIR | None = None,
    error_budget: float | None = None,
    physical_error: float | None = None,
) -> FTQCResourceReport:
    """Estimate a declared component without creating a circuit.

    Supply an already emitted circuit for a side-by-side instruction count.
    The caller pairs that circuit with the component model. Supply both error
    rates to estimate a Beverland fast data block from the logical model.
    """
    if (error_budget is None) != (physical_error is None):
        raise ValueError("Supply both error_budget and physical_error")
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
