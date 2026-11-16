"""Query modeled costs of verified typed Program definitions by name."""

from collections.abc import Mapping, MutableMapping, Sequence

from ...core.ir.program import ProgramRecord, verify_program_record
from ..registers.component_expansion import _has_call_decomposition
from .call_graph import Generalizer, compose_cost_generalizer
from .cost_query import CostMetric, Value, _query_cost


def query_program_cost(
    record: ProgramRecord,
    metric: CostMetric[str, Value],
    entry: str = "main",
    *,
    cache: MutableMapping[str, Value] | None = None,
    static_costs: Mapping[str, Value] | None = None,
    generalizer: Generalizer[str] | Sequence[Generalizer[str]] | None = None,
) -> Value:
    """Apply a metric to the typed entry with source composite cache semantics.

    Connected call bodies bypass seeded costs and store only validated static
    costs. Primitive or mixed definitions can reuse cached values. A metric
    owns its callee selection and value algebra.
    """
    diagnostics = verify_program_record(record)
    if diagnostics:
        raise ValueError(f"invalid register Program: {diagnostics}")
    functions = {function.name: function for function in record.functions}
    if (
        record.physical_context is not None
        or entry not in functions
        or functions[entry].registers is None
    ):
        raise ValueError(f"Cost query requires a typed Program entry: {entry!r}")

    def is_composite(name: str) -> bool:
        if name not in functions or functions[name].registers is None:
            raise ValueError(f"unknown typed Program definition {name!r}")
        return _has_call_decomposition(functions[name], functions)

    return _query_cost(
        entry,
        metric.compute,
        metric.zero,
        compose_cost_generalizer(generalizer),
        cache,
        static_costs,
        validate_static=metric.validate_static,
        is_composite=is_composite,
    )
