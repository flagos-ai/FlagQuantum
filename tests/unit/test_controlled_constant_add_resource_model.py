"""Single-controlled AddK source graphs, polarity and original shape fallback."""

from dataclasses import fields, replace

import pytest
import sympy

from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import CliffordCall
from flagquantum.compiler.registers.arithmetic_declarations import arithmetic_call_graph
from tests.unit.test_constant_add_resource_model import CASES
from tests.unit.test_constant_add_resource_model import source as plain_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
CONTROLLED = [
    replace(case, operation="controlled_constant_add", control_value=cv)
    for case in CASES
    for cv in (0, 1)
]


def source(call):
    if not isinstance(call, ArithmeticCall) or call.operation not in (
        "controlled_constant_add",
        "controlled_xor_constant",
    ):
        return plain_source(call)
    from qualtran import CtrlSpec

    operation = (
        "constant_add"
        if call.operation == "controlled_constant_add"
        else "xor_constant"
    )
    plain = replace(call, operation=operation, control_value=1, adjoint=False)
    result = plain_source(plain).controlled(ctrl_spec=CtrlSpec(cvs=call.control_value))
    return result.adjoint() if call.adjoint else result


@pytest.mark.parametrize("root", CONTROLLED)
def test_original_controlled_wrapper_and_each_direct_callee_row(root):
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    rows = arithmetic_call_graph(root)
    for node, row in rows.items():
        assert [(source(child), count) for child, count in row.items()] == list(
            source(node).build_call_graph(None).items()
        )
    if root.control_value == 0:
        assert list(rows[root].items()) == [
            (replace(root, control_value=1), 1),
            (CliffordCall("x", 1), 2),
        ]
    for legacy in (False, True):
        actual = query_qec_gate_cost(root, declared_calls=rows, legacy_shims=legacy)
        expected = get_cost_value(source(root), QECGatesCost(legacy_shims=legacy))
        for field in fields(actual):
            assert (
                sympy.simplify(
                    getattr(actual, field.name) - getattr(expected, field.name)
                )
                == 0
            )


@pytest.mark.parametrize("cv", (0, 1))
def test_existing_query_consumes_replacement_controlled_declarations(cv):
    from flagquantum.compiler._component_trace.resource_generalization import FixedTCall

    root = ArithmeticCall("controlled_constant_add", (3,), constant=5, control_value=cv)
    actual = query_qec_gate_cost(
        root, declared_calls={root: {FixedTCall("fredkin"): 7}}
    )
    assert (
        actual.cswap == 7
        and actual.and_bloq == actual.clifford == actual.measurement == 0
    )
