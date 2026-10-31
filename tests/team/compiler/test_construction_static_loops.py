"""Bounded classical ranges build fresh SSA with explicit scope and budgets."""

from collections.abc import Callable, Iterable
from types import FunctionType

import pytest

from flagquantum.compiler._construction.frontend_python import capture, qfunc
from flagquantum.compiler._construction.lifetimes import analyze_lifetimes
from flagquantum.compiler._construction.profile import verify_construction
from flagquantum.compiler._hybrid.model import HybridProgram
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.team.compiler.test_construction_python import _cell

pytestmark = pytest.mark.unit


def weighted(x: int, y: int, steps: int) -> int:
    total = x
    for i in range(steps):
        term = y * i
        total = total + term
    return total


def predicate(x: int, y: int, steps: int) -> bool:
    flag = x < 3
    for cutoff in range(1, steps + 1):
        flag = flag and (y < cutoff)
    return flag


def returned_bound(graph: HybridProgram) -> int:
    analysis = analyze_lifetimes((graph,), entry=graph.name)
    result = analysis.program.body.blocks[0].operations[-1].operands[0].id
    return analysis.values[result].upper_bound


@pytest.mark.parametrize("steps", [0, 1, 2, 3])
def test_weighted_accumulator_has_exact_range_and_unique_flat_ssa(steps: int) -> None:
    graph = capture(weighted, {"x": 1, "y": 1}, constants={"steps": steps})
    assert returned_bound(graph) == 1 + steps * (steps - 1) // 2
    assert verify_construction((graph,), entry=graph.name) is graph
    block = graph.body.blocks[0]
    values = [*block.arguments, *(v for op in block.operations for v in op.results)]
    assert len({v.id for v in values}) == len(values)
    assert all(not op.regions for op in block.operations)
    assert all(
        op.name in {"arith.constant", "qint.add", "qint.multiply", "qint.return"}
        for op in block.operations
    )
    if steps == 0:
        assert block.operations[-1].operands[0] == block.arguments[0]
        assert all(
            op.name in {"arith.constant", "qint.return"} for op in block.operations
        )


@pytest.mark.parametrize(
    "expression,total",
    [
        ("range(3)", 3),
        ("range(1,4)", 6),
        ("range(1,6,2)", 9),
        ("range(3,1)", 0),
        ("range(0)", 0),
    ],
)
def test_range_forms_and_empty_extents_fold_classical_indices(
    expression: str, total: int
) -> None:
    fn = _cell(
        f"def main(x):\n    total=x*0\n    for i in {expression}:\n        total=total+i\n    return total+x\n"
    )
    assert returned_bound(capture(fn, {"x": 1})) == total + 1


def test_carry_alias_snapshot_is_not_overwritten_and_body_locals_are_per_iteration() -> (
    None
):
    def snapshot(x: int, y: int) -> int:
        total = x
        previous = total
        for i in range(3):
            term = y * i
            total = total + term
        return previous + total

    graph = capture(snapshot, {"x": 1, "y": 1})
    assert returned_bound(graph) == 5
    block = graph.body.blocks[0]
    final_add = block.operations[-2]
    assert final_add.operands[0] == block.arguments[0]
    assert final_add.operands[1] != block.arguments[0]


def test_helper_loops_qfunc_and_classical_loop_bounds_do_not_execute_bodies() -> None:
    def bound(n: int) -> int:
        total = n
        for j in range(2):
            total = total + 1
        return total

    def main(x: int, n: int) -> int:
        total = x
        for i in range(bound(n)):
            total = total + x
        return total

    graph = capture(qfunc(main), {"x": 1}, constants={"n": 1}, functions=(bound,))
    assert returned_bound(graph) == 4
    with pytest.raises(TypeError, match="captured quantum"):
        qfunc(main)(1, 1)


@pytest.mark.parametrize(
    "statement",
    [
        "for i in [0,1]:\n    total=total+x",
        "for i in range():\n    total=total+x",
        "for i in range(1,2,3,4):\n    total=total+x",
        "for i in range(stop=2):\n    total=total+x",
        "for i in range(*x):\n    total=total+x",
        "for i,j in range(2):\n    total=total+x",
        "for i in range(2):\n    total=total+x\nelse:\n    total=x",
        "for i in range(2):\n    break",
        "for i in range(2):\n    continue",
        "for i in range(2):\n    return total",
        "for i in range(2):\n    for j in range(2):\n        total=total+x",
    ],
)
def test_unadmitted_loop_statements_refuse_at_source(statement: str) -> None:
    text = (
        "def main(x):\n    total=x\n"
        + "\n".join("    " + s for s in statement.splitlines())
        + "\n    return total\n"
    )
    with pytest.raises(
        HybridVerificationError, match="construction.python.loop"
    ) as error:
        capture(_cell(text), {"x": 1})
    assert "<cs04-cached-cell>:" in str(error.value)


@pytest.mark.parametrize(
    "body",
    [
        "for i in range(x):\n    total=total+x",
        "for i in range(0,3,0):\n    total=total+x",
        "for i in range(-1):\n    total=total+x",
        "for x in range(2):\n    total=total+x",
        "for total in range(2):\n    total=total+x",
        "for i in range(2):\n    i=i+1",
        "for i in range(2):\n    x=x+1",
        "for i in range(2):\n    total=total+x\n    total=total+x",
        "for i in range(2):\n    temp=x\n    temp=temp+x",
    ],
)
def test_range_quantum_control_shadowing_and_readonly_rebinding_refuse(
    body: str,
) -> None:
    text = (
        "def main(x):\n    total=x\n"
        + "\n".join("    " + s for s in body.splitlines())
        + "\n    return total\n"
    )
    with pytest.raises(HybridVerificationError, match="construction.python"):
        capture(_cell(text), {"x": 1})


@pytest.mark.parametrize("use", ["i", "temp"])
def test_index_and_new_loop_locals_cannot_escape(use: str) -> None:
    fn = _cell(
        f"def main(x):\n    total=x\n    for i in range(2):\n        temp=x+i\n        total=total+temp\n    return {use}\n"
    )
    with pytest.raises(HybridVerificationError, match="construction.python.name"):
        capture(fn, {"x": 1})


@pytest.mark.parametrize("expression", ["missing(x)", "x.mutate()", "not x"])
def test_empty_loops_validate_hidden_calls_and_types(expression: str) -> None:
    fn = _cell(
        f"def main(x):\n    total=x\n    for i in range(0):\n        total={expression}\n    return total\n"
    )
    with pytest.raises(HybridVerificationError, match="construction.python"):
        capture(fn, {"x": 2})


def test_empty_validation_rolls_back_value_ids_without_leaking_provisional_work() -> (
    None
):
    fn = _cell(
        "def main(x):\n    total=x\n    for i in range(0):\n        total=total+x\n    tail=total+1\n    return tail\n"
    )
    graph = capture(fn, {"x": 1})
    block = graph.body.blocks[0]
    assert [op.name for op in block.operations].count("qint.add") == 1
    assert returned_bound(graph) == 2
    assert verify_construction((graph,), entry=graph.name) is graph


def test_single_active_loop_applies_across_registered_helpers_even_for_empty_range() -> (
    None
):
    def inner(x: int) -> int:
        total = x
        for j in range(2):
            total = total + x
        return total

    for extent in (0, 1):
        fn = _cell(
            f"def main(x):\n    total=x\n    for i in range({extent}):\n        total=inner(total)\n    return total\n"
        )
        with pytest.raises(HybridVerificationError, match="loop.nested"):
            capture(fn, {"x": 1}, functions=(inner,))


def test_total_iteration_budget_includes_multiple_loops_and_helpers() -> None:
    def bounded(x: int, n: int) -> int:
        total = x
        for i in range(n):
            total = total + 0
        return total

    assert returned_bound(capture(bounded, {"x": 1}, constants={"n": 256})) == 1
    for n in (257, 2**100):
        with pytest.raises(HybridVerificationError, match="loop.budget"):
            capture(bounded, {"x": 1}, constants={"n": n})
    fn = _cell(
        "def main(x):\n    a=bounded(x,128)\n    b=bounded(a,129)\n    return b\n"
    )
    with pytest.raises(HybridVerificationError, match="loop.budget"):
        capture(fn, {"x": 1}, functions=(bounded,))


def test_expanded_operations_are_bounded_even_with_admitted_iteration_count() -> None:
    assignments = "\n".join(f"        t{i}=total+x" for i in range(20))
    fn = _cell(
        f"def main(x):\n    total=x\n    for i in range(256):\n{assignments}\n    return total\n"
    )
    with pytest.raises(HybridVerificationError, match="budget"):
        capture(fn, {"x": 1})


def test_range_and_registered_helpers_cannot_be_shadowed_by_loop_bindings() -> None:
    def helper(x: int) -> int:
        return x

    snippets = [
        "range=2\nfor i in range(2):\n    total=total+x",
        "for helper in range(2):\n    total=total+x",
        "for i in range(2):\n    helper=x",
        "for i in range(0):\n    range=x",
        "helper=x\nfor i in range(2):\n    total=helper(x)",
    ]
    for snippet in snippets:
        text = (
            "def main(x):\n    total=x\n"
            + "\n".join("    " + s for s in snippet.splitlines())
            + "\n    return total\n"
        )
        with pytest.raises(HybridVerificationError, match="construction.python"):
            capture(_cell(text), {"x": 1}, functions=(helper,))


def test_range_helpers_cannot_hide_quantum_work_behind_a_classical_return() -> None:
    def hidden(x: int) -> int:
        _quantum = x + x
        return 0

    fn = _cell(
        "def main(x):\n    total=x\n    for i in range(hidden(x)):\n        total=total+x\n    return total\n"
    )
    with pytest.raises(HybridVerificationError, match="loop.range"):
        capture(fn, {"x": 1}, functions=(hidden,))


@pytest.mark.parametrize("binding", ["global", "closure"])
def test_overridden_builtin_range_is_rejected_without_calling_it(binding: str) -> None:
    calls: list[int] = []

    def foreign_range(n: int) -> list[int]:
        calls.append(n)
        return []

    if binding == "global":
        function = _cell(
            "def main(x):\n    total=x\n    for i in range(2):\n        total=total+x\n    return total\n"
        )
        assert isinstance(function, FunctionType)
        function.__globals__["range"] = foreign_range
    else:

        def definition(range: Callable[[int], Iterable[int]]) -> Callable[[int], int]:
            def main(x: int) -> int:
                total = x
                for i in range(2):
                    total = total + x
                return total

            return main

        function = definition(foreign_range)
    with pytest.raises(HybridVerificationError, match="loop.range"):
        capture(function, {"x": 1})
    assert calls == []
