"""Query modeled costs of verified typed Program definitions by name."""

from collections.abc import Callable, Mapping, MutableMapping, Sequence

from ...core.ir.program import ProgramRecord, verify_program_record
from ..registers.component_expansion import _has_call_decomposition
from .call_graph import Generalizer, compose_cost_generalizer
from .cost_query import CostMetric, Value, _query_cost
from .qec_gate_cost import QECGateCost, _validate_static_gate_cost, query_qec_gate_cost
from .resource_generalization import ResourceCall


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


def query_program_qec_gate_cost(
    record: ProgramRecord,
    terminal_keys: Mapping[str, ResourceCall],
    entry: str = "main",
    *,
    legacy_shims: bool = False,
    cache: MutableMapping[str, QECGateCost] | None = None,
    static_costs: Mapping[str, QECGateCost] | None = None,
    generalizer: Generalizer[str] | Sequence[Generalizer[str]] | None = None,
) -> QECGateCost:
    """Sum modeled QEC costs over stored calls with composite cache semantics.

    Call-only bodies expand through the verified Program. Primitive and
    mixed bodies require a declared terminal key and use the existing QEC
    query for leaf classification. No gate emission or binding is performed.
    """
    functions = {function.name: function for function in record.functions}
    leaf_cache: dict[ResourceCall, QECGateCost] = {}

    def compute(name: str, child: Callable[[str], QECGateCost]) -> QECGateCost:
        function = functions[name]
        if not _has_call_decomposition(function, functions):
            if name not in terminal_keys:
                raise ValueError(
                    f"Missing QEC terminal key for Program definition {name!r}"
                )
            return query_qec_gate_cost(
                terminal_keys[name], legacy_shims=legacy_shims, cache=leaf_cache
            )
        total = QECGateCost()
        for operation in function.blocks[0].operations:
            assert operation.callee is not None
            total += child(operation.callee)
        return total

    return query_program_cost(
        record,
        CostMetric(compute, QECGateCost(), _validate_static_gate_cost),
        entry,
        cache=cache,
        static_costs=static_costs,
        generalizer=generalizer,
    )
