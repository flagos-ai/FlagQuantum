"""One-bit predicates in the existing closed unsigned construction profile."""

from dataclasses import replace

import pytest

from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.lifetimes import analyze_lifetimes
from flagquantum.compiler._construction.profile import uint_type, verify_construction
from flagquantum.compiler._hybrid.model import (
    INDEX,
    Block,
    HybridProgram,
    Operation,
    Region,
    Value,
)
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.team.compiler.test_construction_python import _cell

pytestmark = pytest.mark.unit


def predicate(x: int, y: int) -> bool:
    return x * x + y < 5


def equal(x: int, y: int) -> bool:
    return x == y


@pytest.mark.parametrize(
    "function,opcode", [(predicate, "qint.less"), (equal, "qint.equal")]
)
def test_capture_and_inlining_keep_one_bit_range_and_dependencies(function, opcode):
    graph = capture(function, {"x": 2, "y": 2})
    assert graph.body.blocks[0].operations[-1].operands[0].type == uint_type(1)
    analysis = analyze_lifetimes((graph,), entry=graph.name)
    comparison = next(
        op for op in analysis.program.body.blocks[0].operations if op.name == opcode
    )
    value = analysis.values[comparison.results[0].id]
    assert value.upper_bound == 1
    assert value.dependencies
    assert comparison.location.filename.endswith(__file__)
    assert verify_construction((graph,), entry=graph.name) is graph


def test_explicit_bool_helper_and_numeric_use_compose_without_new_types():
    def main(x: int, y: int) -> int:
        flag = equal(x, y)
        return flag + x

    graph = capture(main, {"x": 2, "y": 2}, functions=(equal,))
    assert any(op.name == "qint.equal" for op in graph.body.blocks[0].operations)
    assert graph.body.blocks[0].operations[-1].operands[0].type == uint_type(3)


def test_constant_comparisons_fold_exact_values_not_upper_bound_comparison():
    graph = capture(
        _cell("def main(x):\n    return x + (2 < 1) + (2 == 2)\n"), {"x": 2}
    )
    constants = [
        op.attributes["value"]
        for op in graph.body.blocks[0].operations
        if op.name == "arith.constant"
    ]
    assert 0 in constants and 1 in constants
    assert not any(
        op.name in {"qint.equal", "qint.less"} for op in graph.body.blocks[0].operations
    )


@pytest.mark.parametrize(
    "body",
    [
        "x is y",
        "x in y",
        "x and y",
        "not x",
        "True == x",
    ],
)
def test_unsupported_conditions_are_early_source_diagnostics(body):
    with pytest.raises(HybridVerificationError, match="construction.python") as error:
        capture(_cell(f"def main(x,y):\n    return {body}\n"), {"x": 2, "y": 2})
    assert "<cs04-cached-cell>:2:" in str(error.value)


def test_bool_parameters_and_non_one_bit_bool_return_are_refused():
    def parameter(x: bool) -> bool:
        return x

    def result(x: int) -> bool:
        return x + 1

    for function in (parameter, result):
        with pytest.raises(HybridVerificationError, match="annotation|one-bit"):
            capture(function, {"x": 2})


@pytest.mark.parametrize("mutation", ["wide", "arity", "attributes", "classical"])
def test_invalid_comparison_records_are_not_trusted(mutation):
    graph = capture(equal, {"x": 2, "y": 2})
    block = graph.body.blocks[0]
    op = block.operations[0]
    if mutation == "wide":
        op = replace(op, results=(Value(op.results[0].id, uint_type(2)),))
    elif mutation == "arity":
        op = replace(op, operands=op.operands[:1])
    elif mutation == "attributes":
        op = replace(op, attributes={"trusted": True})
    else:
        # Two actual classical definitions are checked before the comparison.
        a, b = (Value(v.id, INDEX) for v in block.arguments)
        constants = (
            Operation("arith.constant", results=(a,), attributes={"value": 1}),
            Operation("arith.constant", results=(b,), attributes={"value": 2}),
        )
        op = replace(op, operands=(a, b))
        graph = HybridProgram(
            graph.name, Region((Block((), (*constants, op, block.operations[-1])),))
        )
        with pytest.raises(HybridVerificationError, match="comparison.signature"):
            verify_construction((graph,), entry=graph.name)
        return
    graph = replace(
        graph, body=Region((replace(block, operations=(op, block.operations[-1])),))
    )
    with pytest.raises(HybridVerificationError, match="comparison|attributes"):
        verify_construction((graph,), entry=graph.name)


def test_string_bool_annotation_and_optional_qfunc_share_the_same_profile():
    from flagquantum import synth

    @synth.qfunc
    def decorated(x: int, y: int) -> bool:
        return x < y

    graph = capture(decorated, {"x": 2, "y": 2})
    assert graph.body.blocks[0].operations[-1].operands[0].type == uint_type(1)
    textual = capture(_cell("def main(x) -> 'bool':\n    return x == 1\n"), {"x": 2})
    assert textual.body.blocks[0].operations[-1].operands[0].type == uint_type(1)
