"""Complete sorting hierarchies preserve source history and exact inverses."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.comparison.compare_exchange_component import (
    compare_exchange_program,
)
from flagquantum.compiler.registers.comparison.parallel_comparison import (
    parallel_comparison_pairs,
)
from flagquantum.compiler.registers.comparison.parallel_comparison_component import (
    parallel_comparison_program,
)
from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.compiler.registers.sorting.bitonic_merge_component import (
    bitonic_merge_program,
)
from flagquantum.compiler.registers.sorting.bitonic_sort_component import (
    bitonic_sort_program,
)
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_example_checks import (
    CHECKS,
    CheckKind,
    CheckStatus,
    RegisterExample,
    check_example,
)
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def sorting_case(kind: str, width: int = 1) -> tuple[ProgramRecord, Any, int, int]:
    from qualtran.bloqs.arithmetic.sorting import (
        BitonicMerge,
        BitonicSort,
        Comparator,
        ParallelComparators,
    )

    if kind == "compare":
        return compare_exchange_program(width), Comparator(width), 2, 1
    if kind == "parallel":
        return (
            parallel_comparison_program(5, 2, width),
            ParallelComparators(5, 2, width),
            5,
            2,
        )
    if kind == "merge":
        return bitonic_merge_program(2, width), BitonicMerge(2, width), 4, 4
    return bitonic_sort_program(4, width), BitonicSort(4, width), 4, 6


def inverse_graph(kind: str, width: int = 1) -> ProgramRecord:
    if kind == "compare":
        return compare_exchange_program(width, adjoint=True)
    if kind == "parallel":
        return parallel_comparison_program(5, 2, width, adjoint=True)
    if kind == "merge":
        return bitonic_merge_program(2, width, adjoint=True)
    return bitonic_sort_program(4, width, adjoint=True)


def bind_one_bit_case(
    kind: str,
) -> tuple[BoundRegisterProgram, BoundRegisterProgram, int, int]:
    graph, _, words, histories = sorting_case(kind)
    pool = tuple(range(words, words + histories + 2))
    inputs = tuple((i,) for i in range(words))
    forward = bind_register_program(
        graph, inputs, clean=pool, allow_workspace_outputs=True
    )
    assert forward.outputs == tuple((i,) for i in range(words + histories))
    backward = bind_register_program(
        inverse_graph(kind),
        forward.outputs,
        clean=(words + histories, words + histories + 1),
        input_ownership=tuple((i, False) for i in range(words, words + histories)),
    )
    assert backward.outputs == inputs
    return forward, backward, words, histories


@pytest.mark.parametrize("kind", ("compare", "parallel", "merge", "sort"))
@pytest.mark.parametrize("adjoint", (False, True))
def test_source_network_dependencies_and_named_connections(
    kind: str, adjoint: bool
) -> None:
    graph, source, _, _ = sorting_case(kind, 2)
    if adjoint:
        graph, source = inverse_graph(kind, 2), source.adjoint()
    assert verify_program_record(graph) == ()
    terminals = {
        f.name
        for f in graph.functions
        if f.name != "main"
        and (
            tuple(r.name for r in f.registers or ()) == ("a", "b", "out")
            or kind == "compare"
        )
    }
    compare_layers(graph, source, terminals)


@pytest.mark.parametrize("count,offset", tuple(product(range(1, 10), range(1, 10))))
def test_all_original_offset_pair_orders(count: int, offset: int) -> None:
    from qualtran.bloqs.arithmetic.sorting import ParallelComparators

    graph = parallel_comparison_program(count, offset, 1)
    compare_layers(
        graph,
        ParallelComparators(count, offset, 1),
        {f.name for f in graph.functions if f.name != "main"},
    )
    assert len(parallel_comparison_pairs(count, offset)) == sum(
        op.callee is not None for op in graph.functions[0].blocks[0].operations
    )


@pytest.mark.parametrize("width", (1, 2, 3, 5, 8))
@pytest.mark.parametrize("adjoint", (False, True))
def test_comparator_source_gate_stream_feedback_and_cost(
    width: int, adjoint: bool
) -> None:
    from qualtran.bloqs.arithmetic.sorting import Comparator
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    inputs = (tuple(range(width)), tuple(range(width, 2 * width)))
    history = (2 * width,)
    work = tuple(range(2 * width + 1, 5 * width))
    source = Comparator(width)
    if adjoint:
        source = source.adjoint()
        bound = bind_register_program(
            compare_exchange_program(width, adjoint=True),
            (*inputs, history),
            clean=work,
            input_ownership=((history[0], False),),
        )
    else:
        bound = bind_register_program(
            compare_exchange_program(width),
            inputs,
            clean=(*history, *work),
            allow_workspace_outputs=True,
        )
    ports = {"a": inputs[0], "b": inputs[1]} | ({"out": history} if adjoint else {})
    outputs = {"a": inputs[0], "b": inputs[1]} | ({} if adjoint else {"out": history})
    plan = tuple(w for e in bound.events if e.kind == "allocate" for w in e.wires)
    if width == 8:
        # Native dependency layers visit four independent pair mixers before
        # their parents. Original SDK expansion allocates in subtree postorder.
        # Each mixer owns three wires. Logical history and final flags stay fixed.
        offset = 0 if adjoint else 1
        triples = tuple(plan[offset + 3 * i : offset + 3 * i + 3] for i in range(7))
        plan = (
            *plan[:offset],
            *(wire for i in (0, 1, 4, 2, 3, 5, 6) for wire in triples[i]),
            *plan[offset + 21 :],
        )
    expected = reference_arithmetic_gates(
        source.decompose_bloq(),
        ports,
        plan,
        MeasurementSlots(bound.circuit.instructions),
        reuse_allocations=False,
        preserve_components=True,
        expected_outputs=outputs,
    )
    assert_source(bound.circuit.instructions, expected)
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert counts["t"] + counts["tdg"] == model.total_t_count(ts_per_cswap=7)
    assert counts["measure"] == model.measurement == 2 * width - 1


@pytest.mark.parametrize("kind", ("compare", "parallel", "merge", "sort"))
def test_full_complex_source_isometry_every_branch_and_inverse(kind: str) -> None:
    _, source, words, histories = sorting_case(kind)
    forward, backward, _, _ = bind_one_bit_case(kind)
    raw = torch.arange(1, (1 << (words + 1)) + 1, dtype=torch.float64)
    amps = torch.complex(raw, 7 - 3 * raw)
    amps /= amps.norm()
    # The last logical qubit is an entangled spectator. All history/work start zero.
    total = words + histories + 3
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    initial[:: 1 << (histories + 3)] = amps[::2]
    initial[1 :: 1 << (histories + 3)] = amps[1::2]
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        1 << (words + histories), 1 << words
    )
    transformed = matrix @ amps.reshape(-1, 2)
    expected[::8] = transformed[:, 0]
    expected[1::8] = transformed[:, 1]
    for bound, before, after in (
        (forward, initial, expected),
        (backward, expected, initial),
    ):
        events = sum(g.name == "measure" for g in bound.circuit.instructions)
        assert events == histories
        for branch in product((0, 1), repeat=events):
            actual = forced_branch(before, bound.circuit.instructions, branch)
            torch.testing.assert_close(
                actual, after / 2 ** (events / 2), rtol=0, atol=3e-12
            )


@pytest.mark.integration
def test_append_sorting_consumers() -> None:
    from examples.algorithms.register_bitonic_sort import main as sorting_main
    from examples.algorithms.register_compare_exchange import main as parallel_main

    parallel_main()
    sorting_main()


@pytest.mark.parametrize(
    "factory", ("_comparator", "_parallel_compare", "_bitonic_merge", "_bitonic_sort")
)
@pytest.mark.parametrize("kind", CHECKS)
def test_actual_concrete_factories_existing_modes(
    factory: str, kind: CheckKind
) -> None:
    from qualtran.bloqs.arithmetic import sorting
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.compiler.registers.comparison.compare_exchange import (
        compare_exchange_ir,
    )
    from flagquantum.compiler.registers.comparison.parallel_comparison import (
        parallel_compare_ir,
    )
    from flagquantum.compiler.registers.sorting.bitonic_merge import bitonic_merge_ir
    from flagquantum.compiler.registers.sorting.bitonic_sort import bitonic_sort_ir
    from flagquantum.core.ir import CircuitIR

    source = getattr(sorting, factory).make()
    width = int(source.bitsize)
    if factory == "_comparator":
        graph, count, histories = compare_exchange_program(width), 2, 1
    elif factory == "_parallel_compare":
        count, histories = int(source.k), int(source.num_comparisons)
        graph = parallel_comparison_program(count, int(source.offset), width)
    elif factory == "_bitonic_merge":
        count, histories = 2 * int(source.half_length), int(source.num_comparisons)
        graph = bitonic_merge_program(int(source.half_length), width)
    else:
        count, histories = int(source.k), int(source.num_comparisons)
        graph = bitonic_sort_program(count, width)
    words = tuple(tuple(range(i * width, (i + 1) * width)) for i in range(count))
    history = tuple(range(count * width, count * width + histories))
    work = tuple(
        range(count * width + histories, count * width + histories + 3 * width - 1)
    )
    bits = tuple(range(2 * width - 1))
    if factory == "_comparator":
        original = compare_exchange_ir(words[0], words[1], history[0], work, bits)
    elif factory == "_parallel_compare":
        original = parallel_compare_ir(words, int(source.offset), history, work, bits)
    elif factory == "_bitonic_merge":
        original = bitonic_merge_ir(words, history, work, bits)
    else:
        original = bitonic_sort_ir(words, history, work, bits)

    def lower(record: ProgramRecord) -> CircuitIR:
        return bind_register_program(
            record, words, clean=(*history, *work), allow_workspace_outputs=True
        ).circuit

    bound = lower(graph)
    counts = Counter(g.name for g in bound.instructions)
    assert counts == Counter(g.name for g in original)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count(ts_per_cswap=7)
    assert counts["measure"] == model.measurement
    result = check_example(
        RegisterExample(
            lambda: graph, lower, lambda: Counter(g.name for g in original)
        ),
        kind,
    )
    assert result.status is CheckStatus.PASS, result.message


@pytest.mark.parametrize("adjoint", (False, True))
def test_empty_networks_and_invalid_concrete_shapes(adjoint: bool) -> None:
    for graph in (
        bitonic_sort_program(1, 1, adjoint=adjoint),
        parallel_comparison_program(1, 2, 1, adjoint=adjoint),
    ):
        assert bind_register_program(graph, ((0,),)).circuit.instructions == ()
    with pytest.raises(ValueError):
        bitonic_sort_program(3, 1)
    with pytest.raises(ValueError):
        bitonic_merge_program(3, 1)
    with pytest.raises(ValueError):
        compare_exchange_program(0)
    invalid_direction: Any = 1
    with pytest.raises(ValueError, match="boolean"):
        compare_exchange_program(1, adjoint=invalid_direction)


def source_network_stream(
    source: Any, bound: BoundRegisterProgram, inputs: tuple[tuple[int, ...], ...]
) -> tuple[Instruction, ...]:
    """Read source connections and bind each comparator's actual workspace lifetime."""
    from qualtran.bloqs.arithmetic.sorting import Comparator

    from flagquantum.compiler.registers.comparison.compare_exchange import (
        compare_exchange_ir,
    )
    from tests.unit.register_source_graph import (
        SourcePorts,
        SourceResult,
        walk_source_graph,
    )

    def ports(registers: Any, words: tuple[tuple[int, ...], ...]) -> SourcePorts:
        result = {}
        position = 0
        for register in registers:
            count = int(register.shape[0]) if register.shape else 1
            result[register.name] = tuple(
                w for word in words[position : position + count] for w in word
            )
            position += count
        assert position == len(words)
        return result

    cursor = 0
    slots = MeasurementSlots(bound.circuit.instructions)

    def leaf(bloq: Any, old: SourcePorts, desired: SourcePorts) -> SourceResult | None:
        nonlocal cursor
        underlying = getattr(bloq, "subbloq", bloq)
        if not isinstance(underlying, Comparator):
            return None
        width = int(underlying.bitsize)
        # Exact accepted comparator recipe determines segment size only. Every
        # reference gate below comes from the untouched source decomposition.
        n = len(
            compare_exchange_ir(
                tuple(range(width)),
                tuple(range(width, 2 * width)),
                2 * width,
                tuple(range(2 * width + 1, 5 * width)),
                tuple(range(2 * width - 1)),
            )
        )
        allocations = tuple(
            w
            for event in bound.events
            if event.kind == "allocate"
            and event.instruction_offset is not None
            and cursor <= event.instruction_offset < cursor + n
            for w in event.wires
        )
        outputs = {
            r.name: old[r.name] for r in bloq.signature.rights() if r.name in old
        }
        outputs.update(desired)
        gates = reference_arithmetic_gates(
            bloq.decompose_bloq(),
            old,
            allocations,
            slots,
            reuse_allocations=False,
            preserve_components=True,
            expected_outputs=outputs,
        )
        assert len(gates) == n
        cursor += n
        return gates, outputs

    gates, _ = walk_source_graph(
        source,
        ports(source.signature.lefts(), inputs),
        ports(source.signature.rights(), bound.outputs),
        leaf,
    )
    assert cursor == len(bound.circuit.instructions)
    return gates


@pytest.mark.parametrize("kind", ("parallel", "merge", "sort"))
@pytest.mark.parametrize("width", (1, 2, 3))
@pytest.mark.parametrize("adjoint", (False, True))
def test_full_network_source_primitive_dependencies_and_event_feedback(
    kind: str, width: int, adjoint: bool
) -> None:
    graph, source, count, histories = sorting_case(kind, width)
    inputs = tuple(tuple(range(i * width, (i + 1) * width)) for i in range(count))
    pool = tuple(range(count * width, count * width + histories + 3 * width - 1))
    bound = bind_register_program(
        graph, inputs, clean=pool, allow_workspace_outputs=True
    )
    if adjoint:
        inputs = bound.outputs
        live = {w for word in inputs for w in word}
        history = tuple(w for word in inputs[count:] for w in word)
        bound = bind_register_program(
            inverse_graph(kind, width),
            inputs,
            clean=tuple(w for w in pool if w not in live),
            input_ownership=tuple((w, False) for w in history),
        )
        source = source.adjoint()
    assert_source(
        bound.circuit.instructions, source_network_stream(source, bound, inputs)
    )


@pytest.mark.parametrize(
    "merge,count",
    [(True, k) for k in (2, 4, 8, 16)] + [(False, k) for k in (1, 2, 4, 8, 16)],
)
def test_original_seeded_lists_native_history_work_and_restore(
    merge: bool, count: int
) -> None:
    np = pytest.importorskip("numpy")
    from qualtran.bloqs.arithmetic.sorting import BitonicMerge, BitonicSort

    from tests.unit.measurement_checks import certified_and_channel_map

    width = 3
    source = BitonicMerge(count, width) if merge else BitonicSort(count, width)
    graph = (
        bitonic_merge_program(count, width)
        if merge
        else bitonic_sort_program(count, width)
    )
    total = 2 * count if merge else count
    histories = int(source.num_comparisons)
    words = tuple(tuple(range(i * width, (i + 1) * width)) for i in range(total))
    pool = tuple(range(total * width, total * width + histories + 3 * width - 1))
    bound = bind_register_program(
        graph, words, clean=pool, allow_workspace_outputs=True
    )
    rng = np.random.default_rng(1024)
    if merge:
        samples = [
            tuple(sorted(a.tolist()) + sorted(b.tolist()))
            for a, b in (rng.integers(0, 8, size=(2, count)) for _ in range(5))
        ]
    else:
        samples = [tuple(rng.integers(0, 8, size=count).tolist()) for _ in range(5)]
    before = tuple(
        tuple(
            (value >> (width - 1 - bit)) & 1 for value in values for bit in range(width)
        )
        + (0,) * len(pool)
        for values in samples
    )
    actual, events = certified_and_channel_map(bound.circuit.instructions, before)
    assert events == histories * (2 * width - 1)
    for values, observed in zip(samples, actual, strict=True):
        kwargs = {"xs": np.array(values[:count])}
        if merge:
            kwargs["ys"] = np.array(values[count:])
        result, junk = source.call_classically(**kwargs)
        expected = [0] * bound.circuit.n_wires
        for output, value in zip(bound.outputs, (*result, *junk), strict=True):
            for bit, wire in enumerate(output):
                expected[wire] = (int(value) >> (len(output) - 1 - bit)) & 1
        assert observed == tuple(expected)
    live = {w for word in bound.outputs for w in word}
    junk_wires = tuple(w for word in bound.outputs[total:] for w in word)
    inverse = (
        bitonic_merge_program(count, width, adjoint=True)
        if merge
        else bitonic_sort_program(count, width, adjoint=True)
    )
    backward = bind_register_program(
        inverse,
        bound.outputs,
        clean=tuple(w for w in pool if w not in live),
        input_ownership=tuple((w, False) for w in junk_wires),
    )
    restored, inverse_events = certified_and_channel_map(
        backward.circuit.instructions, actual
    )
    assert restored == before and inverse_events == events
