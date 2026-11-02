"""Complete-domain arithmetic equivalence, sound masks and bounded sharing."""

import math
from collections.abc import Callable
from dataclasses import replace
from itertools import product

import pytest

from benchmarks.synthesis.expressions import EDGES, EXPRESSIONS
from benchmarks.synthesis.oracle import evaluate
from flagquantum.compiler._construction.arithmetic_rewrites import normalize_arithmetic
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.lifetimes import analyze_lifetimes
from flagquantum.compiler._construction.profile import uint_type
from flagquantum.compiler._hybrid.model import (
    Block,
    HybridProgram,
    Operation,
    Region,
    Value,
    ValueId,
)
from flagquantum.compiler._hybrid.verifier import HybridVerificationError

pytestmark = pytest.mark.unit


def _run(graph: HybridProgram, inputs: tuple[int, ...]) -> dict[ValueId, int]:
    block = graph.body.blocks[0]
    values = dict(zip((v.id for v in block.arguments), inputs, strict=True))
    for op in block.operations:
        args = tuple(values[v.id] for v in op.operands)
        if op.name == "qint.return":
            break
        if op.name == "arith.constant":
            answer = op.attributes["value"]
        elif op.name == "qint.add":
            answer = sum(args)
        elif op.name == "qint.multiply":
            answer = math.prod(args)
        elif op.name == "qint.equal":
            answer = int(args[0] == args[1])
        elif op.name == "qint.less":
            answer = int(args[0] < args[1])
        elif op.name == "qint.select":
            answer = args[1] if args[0] else args[2]
        else:
            raise AssertionError(op.name)
        values[op.results[0].id] = answer
    return values


@pytest.mark.parametrize("expression", EXPRESSIONS + EDGES, ids=lambda e: e.name)
@pytest.mark.parametrize("width", [1, 2, 3])
def test_complete_domains_masks_widths_and_idempotence(expression, width: int) -> None:
    graph = capture(expression.function, dict.fromkeys(expression.arguments, width))
    analysis = analyze_lifetimes((graph,), entry=graph.name)
    result = normalize_arithmetic(analysis)
    again = normalize_arithmetic(
        analyze_lifetimes((result.program,), entry=result.program.name)
    )
    assert result.program == again.program
    assert result.facts == again.facts
    assert result.original == analysis.program
    assert (
        result.program.body.blocks[0].arguments
        == analysis.program.body.blocks[0].arguments
    )
    returned = result.program.body.blocks[0].operations[-1].operands[0]
    assert (
        returned.type == analysis.program.body.blocks[0].operations[-1].operands[0].type
    )
    for values in product(range(1 << width), repeat=len(expression.arguments)):
        environment = _run(result.program, values)
        assert environment[returned.id] == evaluate(expression.name, values)
        for key, fact in result.facts.items():
            value = environment[key]
            assert value <= fact.upper_bound
            assert value & fact.known_zero == 0
            assert value & fact.known_one == fact.known_one
    for arg in graph.body.blocks[0].arguments:
        assert result.facts[arg.id].upper_bound == (1 << width) - 1
        assert result.facts[arg.id].known_zero == result.facts[arg.id].known_one == 0


def repeated(x: int, y: int) -> int:
    a = x + y
    b = y + x
    return a * b


def double(x: int) -> int:
    return x + x


def shifted(x: int) -> int:
    return (x * 8 + 3) ** 2


def identities(x: int, y: int) -> int:
    return (x * 1 + 0) * (y + 0)


def square(x: int, y: int) -> int:
    return x * x


def product_value(x: int, y: int) -> int:
    return x * y


@pytest.mark.parametrize("function", [repeated, double, shifted, identities])
def test_general_rewrites_have_exact_intermediate_bit_facts(
    function: Callable[..., int],
) -> None:
    import inspect

    names = tuple(inspect.signature(function).parameters)
    graph = capture(function, dict.fromkeys(names, 2))
    result = normalize_arithmetic(analyze_lifetimes((graph,), entry=graph.name))
    for inputs in product(range(4), repeat=len(names)):
        values = _run(result.program, inputs)
        output = result.program.body.blocks[0].operations[-1].operands[0].id
        assert values[output] == function(*inputs)
        for key, fact in result.facts.items():
            assert values[key] & fact.known_zero == 0
            assert values[key] & fact.known_one == fact.known_one
    if function is double:
        assert result.facts[output].known_zero & 1
    if function is shifted:
        assert result.facts[output].known_one & 1
        assert result.facts[output].known_zero & 6 == 6


def test_cse_merges_commutative_calls_and_keeps_all_source_locations() -> None:
    graph = capture(repeated, {"x": 2, "y": 2})
    result = normalize_arithmetic(analyze_lifetimes((graph,), entry=graph.name))
    operations = result.program.body.blocks[0].operations
    assert [op.name for op in operations] == [
        "qint.add",
        "qint.multiply",
        "qint.return",
    ]
    assert operations[1].operands[0] == operations[1].operands[1]
    assert operations[1].results[0].id in result.squares
    assert len(result.origins[operations[0].results[0].id]) == 2
    assert all(
        loc.filename.endswith(__file__.split("/")[-1])
        for loc in result.origins[operations[0].results[0].id]
    )


def test_equal_width_inputs_are_not_same_values_or_squares() -> None:
    for function, expected in ((square, True), (product_value, False)):
        graph = capture(function, {"x": 2, "y": 2})
        result = normalize_arithmetic(analyze_lifetimes((graph,), entry=graph.name))
        output = result.program.body.blocks[0].operations[-1].operands[0].id
        assert (output in result.squares) is expected
        assert bool(result.facts[output].known_zero & 2) is expected


def test_declared_wide_outputs_are_not_truncated_by_alias_simplification() -> None:
    graph = capture(identities, {"x": 1, "y": 1})
    analysis = analyze_lifetimes((graph,), entry=graph.name)
    block = analysis.program.body.blocks[0]
    old = block.operations[-2]
    wide = Value(old.results[0].id, uint_type(8))
    operations = (
        *block.operations[:-2],
        replace(old, results=(wide,)),
        Operation("qint.return", (wide,)),
    )
    graph = HybridProgram(graph.name, Region((Block(block.arguments, operations),)))
    result = normalize_arithmetic(analyze_lifetimes((graph,), entry=graph.name))
    returned = result.program.body.blocks[0].operations[-1].operands[0]
    assert returned.type == uint_type(8)
    assert result.facts[returned.id].effective_width == 1
    assert result.facts[returned.id].known_zero & 0xFE == 0xFE


def test_node_budget_and_effect_region_admission_are_fail_closed() -> None:
    graph = capture(repeated, {"x": 2, "y": 2})
    analysis = analyze_lifetimes((graph,), entry=graph.name)
    for count in (1, 2):
        with pytest.raises(HybridVerificationError, match="budget"):
            normalize_arithmetic(analysis, max_nodes=count)
    for count in (0, True):
        with pytest.raises((ValueError, TypeError)):
            normalize_arithmetic(analysis, max_nodes=count)
    with pytest.raises(HybridVerificationError, match="4096"):
        normalize_arithmetic(analysis, max_nodes=4097)
    block = analysis.program.body.blocks[0]
    bad = replace(block.operations[0], regions=(Region((Block((), ()),)),))
    effectful = HybridProgram(
        graph.name, Region((Block(block.arguments, (bad, *block.operations[1:])),))
    )
    with pytest.raises(HybridVerificationError, match="regions"):
        normalize_arithmetic(replace(analysis, program=effectful))


def squared_helper(x: int) -> int:
    return x**2


def repeated_helper(x: int) -> int:
    return squared_helper(x) + squared_helper(x)


def test_inlined_pure_helpers_and_literal_power_share_the_same_square() -> None:
    graph = capture(repeated_helper, {"x": 3}, functions=(squared_helper,))
    result = normalize_arithmetic(analyze_lifetimes((graph,), entry=graph.name))
    assert len(result.squares) == 1
    square_id = next(iter(result.squares))
    # The scalar frontend already inlines these helpers; keep its supplied
    # helper-body location. Explicit SSA call locations are checked separately.
    assert result.origins[square_id]
    assert (
        sum(
            op.name == "qint.multiply"
            for op in result.program.body.blocks[0].operations
        )
        == 1
    )
    for x in range(8):
        values = _run(result.program, (x,))
        assert (
            values[result.program.body.blocks[0].operations[-1].operands[0].id]
            == 2 * x * x
        )


def guarded(x: int, y: int) -> int:
    a = x + y
    b = y * x
    c = y + b
    d = y + a
    return d * c


def test_optional_emission_budget_retains_original_and_normalized_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.compiler._construction import lowering

    graph = capture(repeated, {"x": 1, "y": 1})
    args = graph.body.blocks[0].arguments
    monkeypatch.setattr(lowering, "_NORMALIZATION_PROBE_MAX_OPERATIONS", 0)
    candidate = lowering.schedule_arithmetic(
        (graph,),
        {args[0].id: (0,), args[1].id: (1,)},
        (2, 3, 4),
        entry=graph.name,
        n_qubits=32,
        workspace=tuple(range(5, 32)),
        workspace_state="fresh_zero",
    )
    assert candidate.baseline is not None
    assert candidate.circuit == candidate.baseline.circuit
    assert candidate.normalization is not None
    assert len(candidate.normalization.squares) == 1
    assert candidate.normalized_cost is None
    assert candidate.normalization_errors[0].code == "construction.normalization.budget"
    assert "0 operations" in candidate.normalization_errors[0].message


def test_real_depth_regression_keeps_original_candidate_for_later_selection() -> None:
    from flagquantum.compiler._construction.lowering import (
        _logical_cost,
        schedule_arithmetic,
    )

    graph = capture(guarded, {"x": 2, "y": 2})
    block = graph.body.blocks[0]
    width = block.operations[-1].operands[0].type.parameters[0]
    candidate = schedule_arithmetic(
        (graph,),
        {v.id: tuple(range(i * 2, i * 2 + 2)) for i, v in enumerate(block.arguments)},
        tuple(range(4, 4 + width)),
        entry=graph.name,
        n_qubits=128,
        workspace=tuple(range(4 + width, 128)),
        workspace_state="fresh_zero",
        scheduling="recompute",
    )
    assert candidate.baseline is not None and candidate.normalized_cost is not None
    assert candidate.normalized_cost[0] < candidate.baseline.peak_workspace
    assert candidate.normalized_cost[2] > _logical_cost(candidate.baseline)[2]
    assert candidate.circuit == candidate.baseline.circuit
    assert candidate.normalization is not None
    assert candidate.normalization.program != candidate.normalization.original


def test_explicit_ssa_helper_call_sites_survive_sharing() -> None:
    from flagquantum.compiler._hybrid.model import SourceLocation

    x = Value(ValueId("entry", 0), uint_type(2))
    a = Value(ValueId("entry", 1), uint_type(4))
    b = Value(ValueId("entry", 2), uint_type(4))
    total = Value(ValueId("entry", 3), uint_type(5))
    arg = Value(ValueId("helper", 0), uint_type(2))
    square = Value(ValueId("helper", 1), uint_type(4))
    helper = HybridProgram(
        "helper",
        Region(
            (
                Block(
                    (arg,),
                    (
                        Operation(
                            "qint.multiply",
                            (arg, arg),
                            (square,),
                            location=SourceLocation("user.py", 2, 1),
                        ),
                        Operation("qint.return", (square,)),
                    ),
                ),
            )
        ),
    )
    calls = (
        Operation(
            "qint.call",
            (x,),
            (a,),
            {"callee": "helper"},
            location=SourceLocation("user.py", 10, 1),
        ),
        Operation(
            "qint.call",
            (x,),
            (b,),
            {"callee": "helper"},
            location=SourceLocation("user.py", 11, 1),
        ),
        Operation("qint.add", (a, b), (total,)),
        Operation("qint.return", (total,)),
    )
    graph = HybridProgram("entry", Region((Block((x,), calls),)))
    normalized = normalize_arithmetic(analyze_lifetimes((graph, helper), entry="entry"))
    square_id = next(iter(normalized.squares))
    assert {loc.line for loc in normalized.origins[square_id]} == {2, 10, 11}


def test_optional_flattened_node_budget_keeps_a_valid_original_schedule() -> None:
    from flagquantum.compiler._construction.lowering import schedule_arithmetic
    from flagquantum.compiler._hybrid.model import INDEX

    arguments = tuple(Value(ValueId("wide_entry", i), uint_type(1)) for i in range(254))
    helper_arg = Value(ValueId("constant_helper", 0), uint_type(1))
    constants = tuple(
        Operation(
            "arith.constant",
            results=(Value(ValueId("constant_helper", i + 1), INDEX),),
            attributes={"value": 0},
        )
        for i in range(500)
    )
    helper = HybridProgram(
        "constant_helper",
        Region(
            (
                Block(
                    (helper_arg,), (*constants, Operation("qint.return", (helper_arg,)))
                ),
            )
        ),
    )
    calls = tuple(
        Operation(
            "qint.call",
            (arguments[0],),
            (Value(ValueId("wide_entry", 254 + i), uint_type(1)),),
            {"callee": "constant_helper"},
        )
        for i in range(8)
    )
    graph = HybridProgram(
        "wide_entry",
        Region(
            (
                Block(
                    arguments,
                    (*calls, Operation("qint.return", (calls[-1].results[0],))),
                ),
            )
        ),
    )
    candidate = schedule_arithmetic(
        (graph, helper),
        {value.id: (i,) for i, value in enumerate(arguments)},
        (254,),
        entry="wide_entry",
        n_qubits=256,
        workspace=(255,),
        workspace_state="fresh_zero",
    )
    assert candidate.normalization is None
    assert (
        candidate.normalization_errors
        and "budget" in candidate.normalization_errors[0].code
    )
    assert candidate.peak_workspace == 0
    assert [(gate.name, gate.wires) for gate in candidate.circuit.instructions] == [
        ("cx", (0, 254))
    ]
