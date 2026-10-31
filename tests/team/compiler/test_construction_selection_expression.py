"""Exclusive values preserve branch bounds and inverse dependencies in SSA."""

from dataclasses import replace

import pytest

from flagquantum.compiler._construction.frontend_python import capture, qfunc
from flagquantum.compiler._construction.lifetimes import analyze_lifetimes
from flagquantum.compiler._construction.profile import uint_type, verify_construction
from flagquantum.compiler._hybrid.model import INDEX, Block, Operation, Region, Value
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.team.compiler.test_construction_python import _cell

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "expression,width,bound",
    [
        ("x if p else y", 2, 3),
        ("x if x<y else y", 2, 3),
        ("x+1 if p else y", 3, 4),
        ("0 if p else 7", 3, 7),
        ("p if p else 0", 1, 1),
        ("p if p else p", 1, 1),
        ("(x<y) if p else (x==y)", 1, 1),
        ("x if p else (y if x<y else 0)", 2, 3),
    ],
)
def test_ranges_are_branch_maxima_not_the_sum_and_dependencies_are_unique(
    expression: str, width: int, bound: int
) -> None:
    graph = capture(
        _cell(f"def main(p,x,y):\n    return {expression}\n"),
        {"p": 1, "x": 2, "y": 1},
    )
    assert verify_construction((graph,), entry=graph.name) is graph
    block = graph.body.blocks[0]
    assert block.operations[-1].operands[0].type == uint_type(width)
    assert any(op.name == "qint.select" for op in block.operations)
    assert all(op.location and op.location.line == 2 for op in block.operations)
    analysis = analyze_lifetimes((graph,), entry=graph.name)
    result = analysis.program.body.blocks[0].operations[-1].operands[0]
    lifetime = analysis.values[result.id]
    assert lifetime.upper_bound == bound
    assert len(lifetime.dependencies) == len(set(lifetime.dependencies))
    assert all(
        analysis.values[key].value.type != INDEX for key in lifetime.dependencies
    )


@pytest.mark.parametrize("predicate,selected", [("0", "y"), ("1", "x"), ("2<1", "y")])
def test_classical_selection_checks_both_branches_and_folds_guard(
    predicate: str, selected: str
) -> None:
    graph = capture(
        _cell(f"def main(x,y):\n    return x if {predicate} else y\n"),
        {"x": 2, "y": 1},
    )
    block = graph.body.blocks[0]
    assert block.operations[-1].operands[0].id.scope == selected
    assert not any(op.name == "qint.select" for op in block.operations)


def test_constant_guard_does_not_silently_remove_quantum_branch_work() -> None:
    graph = capture(_cell("def main(x):\n    return x if 1 else x*x\n"), {"x": 1})
    assert any(op.name == "qint.multiply" for op in graph.body.blocks[0].operations)


def test_classical_selected_value_can_compose_and_keep_index_type() -> None:
    graph = capture(_cell("def main(x):\n    return x+(2 if 0 else 3)\n"), {"x": 1})
    assert graph.body.blocks[0].operations[-1].operands[0].type == uint_type(3)


def test_helpers_qfunc_aliases_and_static_loop_composition_do_not_execute_body() -> (
    None
):
    def choose(p: int, x: int, y: int) -> int:
        return x if p else y

    main = _cell(
        "def main(p,x,y,n):\n    total=x\n    old=total\n"
        "    for i in range(n):\n        total=choose(p,total,y)\n"
        "    return total+old\n"
    )
    snapshot = qfunc(main)
    # qfunc snapshots source; Python execution is neither necessary nor permitted.
    main.__code__ = _cell(
        "def main(p,x,y,n):\n    raise RuntimeError('executed')\n"
    ).__code__
    graph = capture(
        snapshot, {"p": 1, "x": 2, "y": 1}, constants={"n": 2}, functions=(choose,)
    )
    assert sum(op.name == "qint.select" for op in graph.body.blocks[0].operations) == 2
    assert graph.body.blocks[0].operations[-1].operands[0].type == uint_type(3)


@pytest.mark.parametrize(
    "expression,code",
    [
        ("x if x else y", "conditional"),
        ("x if 2 else y", "conditional"),
        ("x if True else y", "constant"),
        ("missing(x) if 0 else y", "call"),
        ("x if 1 else missing(y)", "call"),
        ("x if p else -1", "syntax"),
        ("x if p else 1.5", "constant"),
        ("x if p else True", "constant"),
        ("x if p else y/2", "operator"),
    ],
)
def test_unselected_errors_and_non_one_bit_truthiness_are_refused(
    expression: str, code: str
) -> None:
    with pytest.raises(
        HybridVerificationError, match=f"construction.python.{code}"
    ) as error:
        capture(
            _cell(f"def main(p,x,y):\n    return {expression}\n"),
            {"p": 1, "x": 2, "y": 1},
        )
    assert "<cs04-cached-cell>:2:" in str(error.value)


@pytest.mark.parametrize(
    "change", ["arity", "guard", "constant_guard", "width", "attributes", "regions"]
)
def test_verifier_rejects_forged_conditional_operations(change: str) -> None:
    graph = capture(
        _cell("def main(p,x,y):\n    return x if p else y\n"), {"p": 1, "x": 2, "y": 1}
    )
    block = graph.body.blocks[0]
    select = block.operations[-2]
    if change == "arity":
        bad = replace(select, operands=select.operands[:2])
    elif change == "guard":
        bad = replace(select, operands=(block.arguments[1], *select.operands[1:]))
    elif change == "constant_guard":
        const_graph = capture(
            _cell("def main(p,x,y):\n    z=0\n    return x if p else z\n"),
            {"p": 1, "x": 2, "y": 1},
        )
        block = const_graph.body.blocks[0]
        graph = const_graph
        select = block.operations[-2]
        bad = replace(select, operands=(select.operands[-1], *select.operands[1:]))
    elif change == "width":
        bad = replace(select, results=(Value(select.results[0].id, uint_type(1)),))
    elif change == "attributes":
        bad = replace(select, attributes={"trust": True})
    else:
        bad = replace(select, regions=(Region((Block((), ()),)),))
    ops = (*block.operations[:-2], bad, Operation("qint.return", bad.results))
    forged = replace(graph, body=Region((replace(block, operations=ops),)))
    with pytest.raises(HybridVerificationError, match="construction"):
        verify_construction((forged,), entry=forged.name)


def test_statement_branches_remain_rejected() -> None:
    with pytest.raises(HybridVerificationError, match="construction.python.statement"):
        capture(
            _cell(
                "def main(p,x,y):\n    if p:\n        z=x\n    else:\n        z=y\n    return z\n"
            ),
            {"p": 1, "x": 1, "y": 1},
        )


def test_selection_nesting_obeys_existing_expression_depth_budget() -> None:
    expression = "x"
    for _ in range(66):
        expression = f"(x if p else {expression})"
    with pytest.raises(HybridVerificationError, match="construction.python.budget"):
        capture(_cell(f"def main(p,x):\n    return {expression}\n"), {"p": 1, "x": 1})


def test_maximum_branch_width_stays_within_the_existing_cap() -> None:
    graph = capture(
        _cell("def main(p,x):\n    return x if p else 0\n"), {"p": 1, "x": 256}
    )
    assert graph.body.blocks[0].operations[-1].operands[0].type == uint_type(256)
