"""Preserve unresolved source Partition/child/Unpartition as typed calls."""

import pytest

from flagquantum.compiler.registers.layout.auto_partition import AutoPartition
from flagquantum.compiler.registers.layout.component_auto_partition import (
    auto_partition_component_program,
)
from flagquantum.compiler.registers.primitives.always_component import (
    always_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from tests.unit.test_component_auto_partition import source_partition
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_component_structured_control import normalize_trace
from tests.unit.test_subtraction_foreign_file import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("left_only", (False, True))
@pytest.mark.parametrize("width_case", ("n", "2*n+1"))
def test_symbolic_wrapper_retains_complete_source_order_and_connections(
    left_only: bool, width_case: str
) -> None:
    import sympy
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.bitwise import BitwiseNot
    from qualtran.bloqs.bookkeeping import Always

    n = sympy.Symbol("n")
    width = n if width_case == "n" else 2 * n + 1
    child = Always(BitwiseNot(QUInt(width)))
    signature = native_signature(child.signature)
    leaf = FunctionRecord("bitwise_not", (), registers=tuple(signature))
    record = always_component_program(ProgramRecord((leaf,)), leaf.name, "main")
    segments = (("x", width),) if left_only else ((None, 1), ("x", width))
    layout = AutoPartition((("packed", segments),), left_only=left_only)
    source = source_partition(child, layout)
    module = auto_partition_component_program(record, layout)
    assert verify_program_record(module) == ()
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.compiler._component_trace.program_cost_query import (
        query_program_qec_gate_cost,
    )
    from flagquantum.compiler._component_trace.qec_gate_cost import QECGateCost

    static_costs = {"bitwise_not": QECGateCost(clifford=width)}
    for function in module.functions:
        if function.name.startswith("__fq_partition_"):
            static_costs[function.name] = QECGateCost()
    assert (
        query_program_qec_gate_cost(
            module, {}, "packed", static_costs=static_costs
        ).asdict()
        == get_cost_value(source, QECGatesCost()).asdict()
    )
    packed = next(f for f in module.functions if f.name == "packed")
    assert packed.registers == tuple(native_signature(source.signature))
    observer = _source(signature)
    assert normalize_trace(_native_trace(module, "packed")) == normalize_trace(
        _source_trace(observer, source.decompose_bloq())
    )
    calls = packed.blocks[0].operations
    assert len(calls) == (2 if left_only else 3)
    definitions = {f.name: f for f in module.functions}
    for operation, (instance, _, _) in zip(
        calls, source.decompose_bloq().iter_bloqsoqs(), strict=True
    ):
        assert operation.callee is not None
        assert definitions[operation.callee].registers == tuple(
            native_signature(instance.bloq.signature)
        )
        if operation.callee != "main":
            assert definitions[operation.callee].blocks == ()
    with pytest.raises(ValueError, match="concrete|bodies"):
        bind_register_program(module, ((0, 1, 2),), entry="packed")


@pytest.mark.parametrize("unused", (False, True))
@pytest.mark.parametrize("depth", (0, 1, 2))
def test_actual_source_factory_serialization_retains_original_failure(
    unused: bool, depth: int
) -> None:
    from qualtran.bloqs.bookkeeping.auto_partition import (
        _auto_partition,
        _auto_partition_unused,
    )
    from qualtran.serialization.bloq import bloqs_to_proto

    source = (_auto_partition_unused if unused else _auto_partition).make()
    with pytest.raises(ValueError, match="setting an array element with a sequence"):
        bloqs_to_proto(source, max_depth=depth)
