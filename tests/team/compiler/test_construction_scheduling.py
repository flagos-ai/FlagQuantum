"""Lifetimes, physical allocation histories and exact scheduled permutations."""

from collections.abc import Callable
from dataclasses import replace

import pytest

from flagquantum.compiler import compile as compile_program
from flagquantum.compiler import estimate_resources
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.lifetimes import analyze_lifetimes
from flagquantum.compiler._construction.lowering import (
    ArithmeticCandidate,
    Schedule,
    lower_arithmetic,
    schedule_arithmetic,
)
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


def pressure(x: int, y: int) -> int:
    t = x * y
    a = t * x
    u = a * y
    return t + u


def shared(x: int, y: int) -> int:
    t = x + y
    a = t * t
    return a + t


def configured_left(x: int, y: int) -> int:
    return 3 + x * y


def configured_right(x: int, y: int) -> int:
    t = x * 3
    return t + y


def zero_product(x: int, y: int) -> int:
    return x * 0 + y**2


def _image(candidate: ArithmeticCandidate, basis: int) -> int:
    n = candidate.circuit.n_wires
    for instruction in candidate.circuit.instructions:
        if all(basis & (1 << (n - 1 - q)) for q in instruction.wires[:-1]):
            basis ^= 1 << (n - 1 - instruction.wires[-1])
    return basis


@pytest.mark.parametrize(
    "function", [shared, configured_left, configured_right, zero_product]
)
@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_shared_additions_constants_padding_and_zero_products(
    function: Callable[[int, int], int],
    strategy: Schedule,
) -> None:
    graph = capture(function, {"x": 2, "y": 1})
    block = graph.body.blocks[0]
    width = block.operations[-1].operands[0].type.parameters[0]
    io = 3 + width
    candidate = schedule_arithmetic(
        (graph,),
        {block.arguments[0].id: (0, 1), block.arguments[1].id: (2,)},
        tuple(range(3, io)),
        entry=graph.name,
        n_qubits=64,
        workspace=tuple(range(io, 64)),
        workspace_state="fresh_zero",
        scheduling=strategy,
    )
    for x in range(4):
        for y in range(2):
            for target in (0, (1 << width) - 1):
                prefix = (x << 1) | y
                initial = ((prefix << width) | target) << (64 - io)
                expected = ((prefix << width) | (target ^ function(x, y))) << (64 - io)
                assert _image(candidate, initial) == expected


def _candidate(
    width: int, strategy: Schedule, capacity: int = 128
) -> ArithmeticCandidate:
    graph = capture(pressure, {"x": width, "y": width})
    block = graph.body.blocks[0]
    result_width = block.operations[-1].operands[0].type.parameters[0]
    begin = 2 * width + result_width
    return schedule_arithmetic(
        (graph,),
        {
            v.id: tuple(range(i * width, (i + 1) * width))
            for i, v in enumerate(block.arguments)
        },
        tuple(range(2 * width, begin)),
        entry=graph.name,
        n_qubits=capacity,
        workspace=tuple(range(begin, capacity)),
        workspace_state="fresh_zero",
        scheduling=strategy,
    )


def test_lifetimes_distinguish_later_uses_from_inverse_dependencies() -> None:
    graph = capture(pressure, {"x": 2, "y": 2})
    analysis = analyze_lifetimes((graph,), entry=graph.name)
    ops = analysis.program.body.blocks[0].operations
    t, a, u = (op.results[0].id for op in ops[:3])
    assert analysis.values[t].uses == (1, 3)
    assert analysis.values[t].last_use == 3
    assert analysis.values[a].uses == (2,)
    assert analysis.values[a].dependencies == (t, ops[1].operands[1].id)
    assert analysis.values[u].dependencies == (a, ops[2].operands[1].id)
    assert analysis.values[ops[3].results[0].id].upper_bound == 90
    assert all(op.location is not None for op in ops)


@pytest.mark.parametrize(
    "width,keep,clean,keep_ops,clean_ops",
    [
        (1, 6, 5, 40, 58),
        (2, 28, 23, 739, 1091),
        (3, 47, 39, 2714, 4010),
    ],
)
def test_fixed_workload_has_real_workspace_gate_tradeoff(
    width: int,
    keep: int,
    clean: int,
    keep_ops: int,
    clean_ops: int,
) -> None:
    retained = _candidate(width, "retain")
    recomputed = _candidate(width, "recompute")
    assert retained.peak_workspace == keep > clean == recomputed.peak_workspace
    assert (
        len(retained.circuit.instructions)
        == keep_ops
        < clean_ops
        == len(recomputed.circuit.instructions)
    )
    assert retained.recomputations == 0 < recomputed.recomputations
    for candidate in (retained, recomputed):
        block = candidate.lifetimes.program.body.blocks[0]
        io = 2 * width + block.operations[-1].operands[0].type.parameters[0]
        # Re-emitting at the measured capacity succeeds; this is actual binding
        # and allocation, not a lower cost written beside an oversized circuit.
        tight = _candidate(width, candidate.strategy, io + candidate.peak_workspace)
        assert tight.circuit.instructions == candidate.circuit.instructions
        assert estimate_resources(tight.circuit).used_qubits == tight.circuit.n_wires
        compiled = estimate_resources(compile_program(tight.circuit))
        assert compiled.used_qubits == tight.circuit.n_wires
        if candidate.strategy == "recompute":
            assert compiled.n_operations > keep_ops
    with pytest.raises(HybridVerificationError, match="zero qubits"):
        _candidate(width, "retain", io + clean)
    for candidate in (retained, recomputed):
        with pytest.raises(HybridVerificationError, match="zero qubits"):
            _candidate(width, candidate.strategy, io + candidate.peak_workspace - 1)


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_allocation_trace_requires_available_inverse_dependencies(
    strategy: Schedule,
) -> None:
    candidate = _candidate(2, strategy)
    inputs = {v.id for v in candidate.lifetimes.program.body.blocks[0].arguments}
    active: dict[ValueId, tuple[int, ...]] = {}
    occupied: set[int] = set()
    peak = 0
    for event in candidate.events:
        lifetime = candidate.lifetimes.values[event.value]
        assert set(lifetime.dependencies) <= set(active) | inputs
        if event.action == "compute":
            assert event.value not in active
            assert not occupied.intersection(event.allocation)
            active[event.value] = event.allocation
            occupied.update(event.allocation)
        else:
            assert active[event.value] == event.allocation
        assert not occupied.intersection(event.scratch)
        assert event.live_workspace == len(occupied) + len(event.scratch)
        peak = max(peak, event.live_workspace)
        if event.action == "uncompute":
            occupied.difference_update(active.pop(event.value))
    assert not active and not occupied
    assert peak == candidate.peak_workspace
    assert candidate == _candidate(2, strategy)


@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_every_small_basis_input_and_nonzero_target_is_an_exact_permutation(
    width: int,
    strategy: Schedule,
) -> None:
    candidate = _candidate(width, strategy)
    result_width = (
        candidate.lifetimes.program.body.blocks[0]
        .operations[-1]
        .operands[0]
        .type.parameters[0]
    )
    io = 2 * width + result_width
    program = _candidate(width, strategy, io + candidate.peak_workspace).circuit
    gates = tuple(
        (
            tuple(1 << (program.n_wires - 1 - q) for q in gate.wires[:-1]),
            1 << (program.n_wires - 1 - gate.wires[-1]),
        )
        for gate in program.instructions
    )
    for x in range(1 << width):
        for y in range(1 << width):
            targets = (
                range(1 << result_width) if width <= 2 else (0, (1 << result_width) - 1)
            )
            for target in targets:
                basis = (
                    ((x << width) | y) << result_width | target
                ) << candidate.peak_workspace
                for controls, bit in gates:
                    if all(basis & control for control in controls):
                        basis ^= bit
                t = x * y
                expected = (
                    ((x << width) | y) << result_width | (target ^ (t + t * x * y))
                ) << candidate.peak_workspace
                assert basis == expected


def _function(
    name: str, args: tuple[Value, ...], ops: tuple[Operation, ...]
) -> HybridProgram:
    return HybridProgram(name, Region((Block(args, ops),)))


def test_function_call_instances_keep_shared_argument_identity_and_distinct_results() -> (
    None
):
    x = Value(ValueId("main", 10), uint_type(1))
    local = Value(ValueId("helper", 0), uint_type(1))
    squared = Value(ValueId("helper", 1), uint_type(1))
    square = _function(
        "square",
        (local,),
        (
            Operation("qint.multiply", (local, local), (squared,)),
            Operation("qint.return", (squared,)),
        ),
    )
    a, b = (Value(ValueId("main", i), uint_type(1)) for i in (11, 12))
    result = Value(ValueId("main", 13), uint_type(2))
    main = _function(
        "main",
        (x,),
        (
            Operation("qint.call", (x,), (a,), {"callee": "square"}),
            Operation("qint.call", (x,), (b,), {"callee": "square"}),
            Operation("qint.add", (a, b), (result,)),
            Operation("qint.return", (result,)),
        ),
    )
    analysis = analyze_lifetimes((main, square), entry="main")
    ops = analysis.program.body.blocks[0].operations
    assert ops[0].operands == ops[1].operands == (x, x)
    assert ops[0].results != ops[1].results
    assert not any(op.name == "qint.call" for op in ops)
    for strategy in ("retain", "recompute"):
        candidate = schedule_arithmetic(
            (main, square),
            {x.id: (0,)},
            (1, 2),
            entry="main",
            n_qubits=16,
            workspace=tuple(range(3, 16)),
            workspace_state="fresh_zero",
            scheduling=strategy,
        )
        assert candidate.lifetimes == analysis
        for value in range(2):
            for target in range(4):
                initial = ((value << 2) | target) << 13
                expected = ((value << 2) | (target ^ (2 * value))) << 13
                assert _image(candidate, initial) == expected


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_mutated_cleanup_dependencies_cycles_and_general_effects_fail_closed(
    strategy: Schedule,
) -> None:
    graph = capture(pressure, {"x": 1, "y": 1})
    block = graph.body.blocks[0]
    x, y = block.arguments
    malformed = [
        (Operation("qint.multiply", (x, y), (x,)), *block.operations),
        (
            Operation(
                "qint.call",
                (x, y),
                (block.operations[0].results[0],),
                {"callee": graph.name},
            ),
            *block.operations[1:],
        ),
        (replace(block.operations[0], name="qint.unitary"), *block.operations[1:]),
        (replace(block.operations[0], name="qint.drop"), *block.operations[1:]),
        (
            replace(block.operations[0], attributes={"clean": True}),
            *block.operations[1:],
        ),
    ]
    for operations in malformed:
        bad = _function(graph.name, block.arguments, tuple(operations))
        with pytest.raises(HybridVerificationError):
            lower_arithmetic(
                (bad,),
                {x.id: (0,), y.id: (1,)},
                (2, 3),
                entry=graph.name,
                n_qubits=16,
                workspace=tuple(range(4, 16)),
                workspace_state="fresh_zero",
                scheduling=strategy,
            )


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_operation_budget_and_workspace_zero_contract_remain_enforced(
    strategy: Schedule,
) -> None:
    graph = capture(pressure, {"x": 1, "y": 1})
    block = graph.body.blocks[0]
    options = dict(
        entry=graph.name,
        n_qubits=16,
        workspace=tuple(range(4, 16)),
        scheduling=strategy,
    )
    bindings = {v.id: (i,) for i, v in enumerate(block.arguments)}
    with pytest.raises(HybridVerificationError, match="fresh or caller-asserted"):
        lower_arithmetic((graph,), bindings, (2, 3), **options)
    with pytest.raises(HybridVerificationError, match="operation budget"):
        lower_arithmetic(
            (graph,),
            bindings,
            (2, 3),
            workspace_state="fresh_zero",
            max_operations=20,
            **options,
        )
    with pytest.raises(HybridVerificationError, match="overlapping"):
        lower_arithmetic(
            (graph,),
            bindings,
            (2, 3),
            entry=graph.name,
            n_qubits=16,
            workspace=(0,),
            workspace_state="fresh_zero",
            scheduling=strategy,
        )


def test_returned_input_and_unused_values_do_not_require_a_consuming_effect() -> None:
    def alias(x: int) -> int:
        _dead = x * x
        return x

    graph = capture(alias, {"x": 1})
    analysis = analyze_lifetimes((graph,), entry=graph.name)
    assert (
        analysis.values[
            analysis.program.body.blocks[0].operations[0].results[0].id
        ].last_use
        is None
    )
    for strategy in ("retain", "recompute"):
        circuit = lower_arithmetic(
            (graph,),
            {graph.body.blocks[0].arguments[0].id: (0,)},
            (1,),
            entry=graph.name,
            n_qubits=4,
            workspace=(2, 3),
            workspace_state="fresh_zero",
            scheduling=strategy,
        )
        assert circuit.instructions


def test_long_recomputation_dependencies_fail_with_a_bounded_diagnostic() -> None:
    x = Value(ValueId("chain", 0), uint_type(1))
    from flagquantum.compiler._hybrid.model import INDEX

    zero = Value(ValueId("chain", 1), INDEX)
    operations = [Operation("arith.constant", results=(zero,), attributes={"value": 0})]
    current = x
    for i in range(70):
        next_value = Value(ValueId("chain", i + 2), uint_type(1))
        operations.append(Operation("qint.multiply", (current, zero), (next_value,)))
        current = next_value
    operations.append(Operation("qint.return", (current,)))
    graph = _function("chain", (x,), tuple(operations))
    options = dict(
        entry="chain",
        n_qubits=96,
        workspace=tuple(range(2, 96)),
        workspace_state="fresh_zero",
    )
    assert lower_arithmetic(
        (graph,), {x.id: (0,)}, (1,), scheduling="retain", **options
    )
    with pytest.raises(HybridVerificationError, match="dependency depth"):
        lower_arithmetic(
            (graph,), {x.id: (0,)}, (1,), scheduling="recompute", **options
        )
