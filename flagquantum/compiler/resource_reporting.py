"""Serialize declared logical models separately from optional emitted IR counts."""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any

from ..core.ir import CircuitIR
from ._component_trace.gate_budget import GateCategories, t_and_ccz_budget
from ._component_trace.resource_comparison import count_emitted_instructions
from ._component_trace.resource_model import ComponentCost, estimate_component_resources
from ._component_trace.symbolic_math import SymbolicInt
from .registers.symbolic_resource_model import SymbolicResourceFormula


def _json_count(value: SymbolicInt) -> int | str:
    # Keep expression text, including unevaluated source formulas; do not round.
    return value if isinstance(value, int) else str(value)


def resource_model_report(
    model: ComponentCost | SymbolicResourceFormula,
    *,
    emitted_ir: CircuitIR | None = None,
    ts_per_rotation: int = 11,
) -> dict[str, Any]:
    """Build a JSON-compatible logical estimate without lowering or execution.

    Concrete components recursively aggregate their declared calls. Symbolic
    formulas retain expressions as strings (not executable serialized code).
    Both use the existing budget rule: by default 11 T per rotation and four T equivalents
    per Toffoli, controlled swap or And. These rates are assumptions, not a
    synthesis result. Raw gate categories are retained alongside that conversion.

    An optional IR is counted syntactically, including conditional instructions
    once each. The caller pairs it with the model; this function does not certify
    semantic equivalence or count the instructions executed by a measured branch.
    No physical hardware, timing, or state simulation is involved.
    """
    if type(ts_per_rotation) is not int or ts_per_rotation < 0:
        raise ValueError("T cost per rotation must be a nonnegative integer")
    gates: GateCategories
    if isinstance(model, ComponentCost):
        estimate = estimate_component_resources(model)
        gates = estimate.gates
        width: SymbolicInt = estimate.peak_logical_qubits
        component = model.name
        success: float | None = estimate.success_probability
    elif isinstance(model, SymbolicResourceFormula):
        gates = model
        width = model.logical_qubits
        component = "register_shape_formula"
        success = None
    else:
        raise TypeError(
            "Resource reporting requires a component model or symbolic formula"
        )
    budget = t_and_ccz_budget(gates, ts_per_rotation=ts_per_rotation)
    emitted = None
    if emitted_ir is not None:
        counts = count_emitted_instructions(emitted_ir.validate())
        emitted = {
            "basis": "emitted_instruction_sequence",
            **asdict(counts),
            "t_count": counts.t_count,
        }
    return {
        "basis": "declared_component_model",
        "component": component,
        "logical_gates": {
            field.name: _json_count(getattr(gates, field.name))
            for field in fields(SymbolicResourceFormula)
            if field.name != "logical_qubits"
        },
        "peak_logical_qubits": _json_count(width),
        "success_probability": success,
        "magic_states": {name: _json_count(value) for name, value in budget.items()},
        "t_equivalents": _json_count(budget["n_t"] + 4 * budget["n_ccz"]),
        "conversion": {
            "t_per_rotation": ts_per_rotation,
            "t_per_ccz_equivalent": 4,
            "ccz_categories": ["toffoli", "cswap", "and_bloq"],
        },
        "expression_encoding": "integer_or_symbolic_expression_string",
        "emitted_ir": emitted,
        "execution_measurements": None,
    }
