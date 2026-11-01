"""Unsigned relation normalization and shared-middle chain semantics."""

import pytest

from flagquantum.compiler._construction.frontend_python import capture, qfunc
from flagquantum.compiler._construction.lifetimes import analyze_lifetimes
from flagquantum.compiler._construction.profile import uint_type, verify_construction
from flagquantum.compiler._hybrid.model import INDEX
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.team.compiler.test_construction_python import _cell

pytestmark = pytest.mark.unit
RELATIONS = ("==", "!=", "<", "<=", ">", ">=")


@pytest.mark.parametrize("relation", RELATIONS)
def test_all_relations_reuse_existing_one_bit_opcodes_and_bounds(relation: str) -> None:
    graph = capture(
        _cell(f"def main(x,y):\n    return x {relation} y\n"), {"x": 2, "y": 1}
    )
    assert verify_construction((graph,), entry=graph.name) is graph
    block = graph.body.blocks[0]
    assert block.operations[-1].operands[0].type == uint_type(1)
    assert {op.name for op in block.operations} <= {
        "qint.less",
        "qint.equal",
        "qint.return",
        "arith.constant",
    }
    assert all(op.location and op.location.line == 2 for op in block.operations)
    analysis = analyze_lifetimes((graph,), entry=graph.name)
    result = analysis.program.body.blocks[0].operations[-1].operands[0]
    assert analysis.values[result.id].upper_bound == 1


@pytest.mark.parametrize(
    "expression",
    [
        "x<=y",
        "x>y",
        "x>=y",
        "x!=y",
        "x<y<3",
        "0<=x<=3",
        "3>=x>y!=0",
        "x==y==x",
        "x<y>x",
        "x<=y>=x",
        "x!=y==0",
        "(0<=x<=3) if y else (x!=y)",
    ],
)
def test_previously_refused_relations_and_mixed_chains_have_one_bit_results(
    expression: str,
) -> None:
    graph = capture(
        _cell(f"def main(x,y):\n    return {expression}\n"), {"x": 2, "y": 1}
    )
    assert graph.body.blocks[0].operations[-1].operands[0].type == uint_type(1)
    assert verify_construction((graph,), entry=graph.name) is graph


@pytest.mark.parametrize("relation", RELATIONS)
@pytest.mark.parametrize("left,right", [(0, 0), (0, 1), (2, 1)])
def test_classical_relations_fold_exact_values(
    relation: str, left: int, right: int
) -> None:
    function = _cell(f"def main(x):\n    return x+({left} {relation} {right})\n")
    expected = function(0)
    graph = capture(function, {"x": 1})
    block = graph.body.blocks[0]
    assert [op.name for op in block.operations] == [
        "arith.constant",
        "qint.add",
        "qint.return",
    ]
    assert block.operations[0].attributes["value"] == expected
    assert block.operations[0].results[0].type == INDEX


@pytest.mark.parametrize(
    "expression", ["0<1<=2!=3>0>=0==0", "3<1<2", "2>1<2", "0!=1==1", "1<=0>=1"]
)
def test_classical_mixed_direction_chains_fold_against_independent_python(
    expression: str,
) -> None:
    function = _cell(f"def main(x):\n    return x+({expression})\n")
    graph = capture(function, {"x": 1})
    block = graph.body.blocks[0]
    assert block.operations[0].attributes["value"] == function(0)
    assert not any(
        op.name in {"qint.less", "qint.equal", "qint.multiply"}
        for op in block.operations
    )


def test_middle_helper_is_captured_once_and_its_original_value_is_shared() -> None:
    def middle(x: int) -> int:
        return x * x

    main = _cell("def main(x):\n    return 3 >= middle(x) > 0\n")
    snapshot = qfunc(main)
    main.__code__ = _cell("def main(x):\n    raise RuntimeError('executed')\n").__code__
    graph = capture(snapshot, {"x": 2}, functions=(middle,))
    ops = graph.body.blocks[0].operations
    squares = [
        op
        for op in ops
        if op.name == "qint.multiply" and op.results[0].type == uint_type(4)
    ]
    assert len(squares) == 1
    square = squares[0].results[0]
    comparisons = [op for op in ops if op.name == "qint.less"]
    assert len(comparisons) == 2
    assert all(square in op.operands for op in comparisons)
    assert all(op.location and op.location.line == 2 for op in comparisons)


@pytest.mark.parametrize(
    "expression,code",
    [
        ("3<1<unknown(x)", "call"),
        ("3<1<-1", "syntax"),
        ("0<x<True", "constant"),
        ("x is y", "comparison"),
        ("x not in y", "comparison"),
        ("0>x is y", "comparison"),
        ("x==y in x", "comparison"),
    ],
)
def test_false_prefixes_do_not_hide_unsupported_source(
    expression: str, code: str
) -> None:
    with pytest.raises(
        HybridVerificationError, match=f"construction.python.{code}"
    ) as error:
        capture(_cell(f"def main(x,y):\n    return {expression}\n"), {"x": 2, "y": 1})
    assert "<cs04-cached-cell>:2:" in str(error.value)


def test_long_chain_respects_expanded_operation_budget_before_lowering() -> None:
    expression = " <= ".join(["x"] * 1100)
    with pytest.raises(HybridVerificationError, match="budget"):
        capture(_cell(f"def main(x):\n    return {expression}\n"), {"x": 1})


def test_false_boolean_prefix_keeps_validated_quantum_chain_work() -> None:
    graph = capture(_cell("def main(x):\n    return 0 and (x < x < 2)\n"), {"x": 1})
    assert sum(op.name == "qint.less" for op in graph.body.blocks[0].operations) == 2
    assert graph.body.blocks[0].operations[-1].operands[0].type == uint_type(1)
