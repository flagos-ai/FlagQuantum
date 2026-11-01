"""One-bit Boolean source semantics reuse the closed unsigned SSA profile."""

import pytest

from flagquantum.compiler._construction.frontend_python import capture, qfunc
from flagquantum.compiler._construction.lifetimes import analyze_lifetimes
from flagquantum.compiler._construction.profile import uint_type, verify_construction
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.team.compiler.test_construction_python import _cell

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "expression",
    [
        "x and y",
        "x or y",
        "not x",
        "not not x",
        "x and x",
        "x or x",
        "x and y and z",
        "x or y or z",
        "x or y and z",
        "(x or y) and not z",
        "(x == y) and not (y == z)",
        "x and 0",
        "0 and x",
        "x or 1",
        "1 or x",
        "x and (1 or 0)",
        "x or (not 1)",
    ],
)
def test_composition_is_one_bit_with_existing_opcodes_and_source_locations(
    expression: str,
) -> None:
    function = _cell(f"def main(x,y,z):\n    return {expression}\n")
    graph = capture(function, {"x": 1, "y": 1, "z": 1})
    assert verify_construction((graph,), entry=graph.name) is graph
    block = graph.body.blocks[0]
    assert block.operations[-1].operands[0].type == uint_type(1)
    assert {op.name for op in block.operations} <= {
        "arith.constant",
        "qint.multiply",
        "qint.equal",
        "qint.return",
    }
    assert all(op.location and op.location.line == 2 for op in block.operations)
    analysis = analyze_lifetimes((graph,), entry=graph.name)
    returned = analysis.program.body.blocks[0].operations[-1].operands[0].id
    assert analysis.values[returned].upper_bound <= 1


@pytest.mark.parametrize(
    "expression,number",
    [
        ("not 0", 1),
        ("not 1", 0),
        ("0 and 1", 0),
        ("1 and 1", 1),
        ("0 or 1 or 0", 1),
        ("0 or 0", 0),
        ("not (1 and 0)", 1),
    ],
)
def test_classical_subexpressions_fold_exact_values_without_new_truthiness(
    expression: str, number: int
) -> None:
    graph = capture(_cell(f"def main(x):\n    return x + ({expression})\n"), {"x": 1})
    constant = graph.body.blocks[0].operations[-2].operands[1]
    definitions = {
        result.id: op for op in graph.body.blocks[0].operations for result in op.results
    }
    assert definitions[constant.id].attributes["value"] == number
    assert not any(op.name == "qint.multiply" for op in graph.body.blocks[0].operations)


@pytest.mark.parametrize(
    "expression",
    [
        "x and y",
        "x or (y == 0)",
        "not x",
        "0 and x",
        "1 or x",
        "(x == 0) and 2",
        "2 or (x == 0)",
        "not 2",
        "(x == 0) or (x+1)",
    ],
)
def test_multibit_and_nonbit_constants_refuse_with_operand_source_location(
    expression: str,
) -> None:
    with pytest.raises(
        HybridVerificationError, match="construction.python.boolean"
    ) as error:
        capture(_cell(f"def main(x,y):\n    return {expression}\n"), {"x": 2, "y": 1})
    assert "<cs04-cached-cell>:2:" in str(error.value)
    assert "one-bit" in str(error.value)


@pytest.mark.parametrize(
    "expression",
    [
        "0 and unavailable(x)",
        "1 or unavailable(x)",
        "0 and x.mutate()",
        "1 or True",
        "not False",
        "0 and (x < x is x)",
    ],
)
def test_constants_do_not_hide_unsupported_source_by_host_short_circuit(
    expression: str,
) -> None:
    with pytest.raises(HybridVerificationError, match="construction.python"):
        capture(_cell(f"def main(x):\n    return {expression}\n"), {"x": 1})


def test_helper_inlining_aliases_and_arithmetic_use_the_same_verified_graph() -> None:
    def helper(x: int, y: int) -> bool:
        return x or not y

    def main(x: int, y: int) -> int:
        flag = helper(x, y)
        alias = flag
        return (flag and alias) + x

    graph = capture(qfunc(main), {"x": 1, "y": 1}, functions=(helper,))
    assert graph.body.blocks[0].operations[-1].operands[0].type == uint_type(2)
    assert verify_construction((graph,), entry=graph.name) is graph
    reused = [
        op
        for op in graph.body.blocks[0].operations
        if op.name == "qint.multiply" and op.operands[0] == op.operands[1]
    ]
    assert reused


@pytest.mark.parametrize("expression", ["~x", "x & y", "x | y", "x ^ y"])
def test_unadmitted_boolean_adjacent_syntax_is_still_refused(expression: str) -> None:
    with pytest.raises(HybridVerificationError, match="construction.python"):
        capture(_cell(f"def main(x,y):\n    return {expression}\n"), {"x": 1, "y": 1})


def test_boolean_expansion_and_nesting_are_bounded_before_lowering() -> None:
    for expression in [" or ".join(["x"] * 900), "not " * 66 + "x"]:
        with pytest.raises(HybridVerificationError, match="construction.python.budget"):
            capture(_cell(f"def main(x):\n    return {expression}\n"), {"x": 1})


def test_conditional_value_is_admitted_without_widening_a_boolean_result() -> None:
    graph = capture(
        _cell("def main(x,y):\n    return x if y else 0\n"), {"x": 1, "y": 1}
    )
    assert graph.body.blocks[0].operations[-1].operands[0].type == uint_type(1)
    assert any(op.name == "qint.select" for op in graph.body.blocks[0].operations)
    assert verify_construction((graph,), entry=graph.name) is graph
