"""Direct phase products keep their exact prefix and omit product allocation."""

from collections.abc import Mapping
from dataclasses import replace

import pytest

from flagquantum.compiler._construction.bindings import OutputMode
from flagquantum.compiler._construction.frontend_python import Definition, capture
from flagquantum.compiler._construction.lowering import (
    ArithmeticCandidate,
    Schedule,
    schedule_arithmetic,
)
from flagquantum.compiler._hybrid.model import HybridProgram, ValueId
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


def mask(enable: int, flag: int) -> bool:
    return enable and flag


def alias(x: int) -> bool:
    return x and x


def controlled(enable: int, x: int, y: int) -> bool:
    return enable and (x < y)


def products(x: int, y: int) -> bool:
    return (x < y) and (y < 3)


def ports(graph: HybridProgram) -> tuple[dict[ValueId, tuple[int, ...]], int]:
    bindings = {}
    position = 0
    for value in graph.body.blocks[0].arguments:
        width = value.type.parameters[0]
        assert isinstance(width, int)
        bindings[value.id] = tuple(range(position, position + width))
        position += width
    return bindings, position


def build(
    function: Definition,
    widths: Mapping[str, int],
    strategy: Schedule,
    *,
    mode: OutputMode = "phase",
    max_operations: int = 100_000,
) -> ArithmeticCandidate:
    graph = capture(function, widths)
    bindings, n_input = ports(graph)
    output = (n_input,) if mode == "value" else ()
    start = n_input + len(output)
    kwargs = dict(
        entry=graph.name,
        workspace_state="fresh_zero",
        scheduling=strategy,
        output_mode=mode,
        max_operations=max_operations,
    )
    probe = schedule_arithmetic(
        (graph,),
        bindings,
        output,
        n_qubits=64,
        workspace=tuple(range(start, 64)),
        **kwargs,
    )
    n = start + probe.peak_workspace
    return schedule_arithmetic(
        (graph,),
        bindings,
        output,
        n_qubits=n,
        workspace=tuple(range(start, n)),
        **kwargs,
    )


def general_phase(
    function: Definition, widths: Mapping[str, int], strategy: Schedule
) -> CircuitIR:
    """Independent general phase baseline from the unchanged value evaluator.

    Its first write to the disjoint value target marks the copy boundary. Keep
    exactly the preceding compute/cleanup prefix, phase its source bit, invert
    that same prefix, and remove the unused value-target slot. The baseline
    retains the value schedule's allocated width; no old source is copied.
    """
    candidate = build(function, widths, strategy, mode="value")
    n_input = sum(widths.values())
    gates = candidate.circuit.instructions
    cut = next(i for i, gate in enumerate(gates) if n_input in gate.wires)
    copy = gates[cut]
    assert copy.name == "cx" and copy.wires[1] == n_input

    def mapped(q: int) -> int:
        assert q != n_input
        return q - 1 if q > n_input else q

    prefix = tuple(
        replace(g, wires=tuple(mapped(q) for q in g.wires)) for g in gates[:cut]
    )
    assert {g.name for g in prefix} <= {"x", "cx", "ccx"}
    instructions = (
        *prefix,
        Instruction("z", (mapped(copy.wires[0]),)),
        *reversed(prefix),
    )
    return CircuitIR(
        n_wires=candidate.circuit.n_wires - 1,
        instructions=instructions,
        dtype="complex128",
    )


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
@pytest.mark.parametrize(
    "function,widths,opcode,wires",
    [
        (mask, {"enable": 1, "flag": 1}, "cz", (0, 1)),
        (alias, {"x": 1}, "z", (0,)),
    ],
)
def test_direct_inputs_and_alias_need_no_product_or_scratch(
    strategy: Schedule,
    function: Definition,
    widths: Mapping[str, int],
    opcode: str,
    wires: tuple[int, ...],
) -> None:
    result = build(function, widths, strategy, max_operations=1)
    assert result.circuit.instructions == (Instruction(opcode, wires),)
    assert result.peak_workspace == 0 and result.events == ()
    assert result.recomputations == 0
    baseline = general_phase(function, widths, strategy)
    assert baseline.n_wires == result.circuit.n_wires + 2
    assert len(baseline.instructions) == 7


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
@pytest.mark.parametrize(
    "function,widths",
    [(controlled, {"enable": 1, "x": 2, "y": 2}), (products, {"x": 2, "y": 2})],
)
def test_both_operands_are_live_at_marker_and_literal_inverse_cleans_every_allocation(
    strategy: Schedule, function: Definition, widths: Mapping[str, int]
) -> None:
    result = build(function, widths, strategy)
    gates = result.circuit.instructions
    middle = len(gates) // 2
    assert gates[middle].name == "cz"
    assert gates[middle + 1 :] == tuple(reversed(gates[:middle]))
    returned = result.lifetimes.program.body.blocks[0].operations[-1].operands[0].id
    assert all(e.value != returned for e in result.events)
    events = result.events
    assert sum(e.operation_count for e in events) + 1 == len(gates)
    live = set()
    for i, event in enumerate(events):
        if i == len(events) // 2:
            for wire in gates[middle].wires:
                assert wire < sum(widths.values()) or wire in live
        if event.action == "compute":
            assert not live.intersection(event.allocation)
            live.update(event.allocation)
        else:
            assert set(event.allocation) <= live
        assert not live.intersection(event.scratch)
        assert len(live) + len(event.scratch) == event.live_workspace
        if event.action == "uncompute":
            live.difference_update(event.allocation)
    assert not live
    assert result.peak_workspace == max(e.live_workspace for e in events)
    baseline = general_phase(function, widths, strategy)
    assert len(gates) < len(baseline.instructions)


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_inverse_and_marker_count_against_the_exact_operation_budget(
    strategy: Schedule,
) -> None:
    result = build(controlled, {"enable": 1, "x": 2, "y": 2}, strategy)
    count = len(result.circuit.instructions)
    assert (
        len(
            build(
                controlled,
                {"enable": 1, "x": 2, "y": 2},
                strategy,
                max_operations=count,
            ).circuit.instructions
        )
        == count
    )
    with pytest.raises(HybridVerificationError, match="operations"):
        build(
            controlled,
            {"enable": 1, "x": 2, "y": 2},
            strategy,
            max_operations=count - 1,
        )


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_classical_wide_and_nonterminal_products_keep_the_general_path(
    strategy: Schedule,
) -> None:
    def mixed(x: int) -> int:
        return x * 0

    def nonterminal(x: int, y: int) -> bool:
        flag = x and y
        _unused = x + y
        return flag

    for function, widths in [(mixed, {"x": 2}), (nonterminal, {"x": 1, "y": 1})]:
        result = build(function, widths, strategy)
        assert (
            result.circuit.instructions[len(result.circuit.instructions) // 2].name
            == "z"
        )
        returned = result.lifetimes.program.body.blocks[0].operations[-1].operands[0].id
        assert any(e.value == returned and e.action == "compute" for e in result.events)
        assert result.peak_workspace > 0


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_value_xor_still_materializes_and_copies_the_product(
    strategy: Schedule,
) -> None:
    result = build(mask, {"enable": 1, "flag": 1}, strategy, mode="value")
    assert result.peak_workspace == 2
    assert any(g.name == "cx" and g.wires[-1] == 2 for g in result.circuit.instructions)
    assert not any(g.name in {"z", "cz"} for g in result.circuit.instructions)
