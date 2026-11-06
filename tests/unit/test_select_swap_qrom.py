"""Pinned SelectSwapQROM graph and complete clean/borrowed bank channels."""

from collections import Counter
from dataclasses import asdict
from itertools import product
from math import prod
from typing import Any

import pytest
import torch

from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.control_predicate import ControlRegister
from flagquantum.compiler.registers.qroam_resource_model import (
    source_select_swap_qrom_cost,
)
from flagquantum.compiler.registers.select_swap_qrom import (
    control_select_swap_qrom_ir,
    optimal_select_swap_log_block_sizes,
    select_swap_qrom_ir,
)
from flagquantum.compiler.registers.source_qroam_clean import QroamCleanTable
from flagquantum.compiler.registers.workspace import RegisterWorkspace
from flagquantum.core.ir import CircuitIR, Instruction
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    source_control_predicate_leaf,
    walk_source_graph,
)
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_relative_controlled_swap import (
    assert_source_dependencies,
    source_relative_swap,
)
from tests.unit.test_source_qrom import source_qrom_gates

pytestmark = pytest.mark.unit


def original_select_swap_gates(
    source: Any,
    ports: SourcePorts,
    banks: tuple[tuple[tuple[int, ...], ...], ...],
    work: tuple[int, ...],
    bits: tuple[int, ...],
    *,
    adjoint: bool = False,
    outer_values: tuple[int, ...] = (),
    predicate_workspace: tuple[int, ...] = (),
    predicate_bits: tuple[int, ...] = (),
) -> tuple[tuple[Instruction, ...], tuple[tuple[Instruction, ...], ...]]:
    pytest.importorskip("qualtran")
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    from qualtran import Controlled, CtrlSpec
    from qualtran.bloqs.arithmetic import Xor
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.bookkeeping import Allocate, Free, Partition
    from qualtran.bloqs.data_loading.qrom import QROM
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox

    words = tuple(word for bank in banks for word in bank)
    allocate = iter(words)
    freed: list[tuple[int, ...]] = []
    inverse_graph = False
    loads = []

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, Allocate):
            word = next(allocate)
            assert bloq.dirty is (source.use_dirty_ancilla and not inverse_graph)
            assert len(word) == bloq.dtype.num_qubits
            return (), {"reg": word}
        if isinstance(bloq, Free):
            # Forward uses the default Free flag even for a borrowed bank.
            # Adjoint exchanges the original Allocate and Free flags.
            assert bloq.dirty is (source.use_dirty_ancilla and inverse_graph)
            freed.append(inputs["reg"])
            return (), {}
        if isinstance(bloq, Partition):
            arrays = {
                reg.name: np.array(
                    [cirq.LineQubit(wire) for wire in inputs[reg.name]], dtype=object
                ).reshape((*reg.shape, int(reg.bitsize)))
                for reg in bloq.signature.lefts()
            }
            operation, returned = bloq.as_cirq_op(qubit_manager=None, **arrays)
            assert operation is None
            physical = {
                name: tuple(w.x for w in value.flat) for name, value in returned.items()
            }
            assert all(physical[name] == wires for name, wires in desired.items())
            return (), physical
        if isinstance(bloq, XGate):
            gates = reference_arithmetic_gates(bloq.as_composite_bloq(), inputs, (), {})
            return gates, inputs
        predicate = source_control_predicate_leaf(
            bloq, inputs, predicate_workspace, predicate_bits
        )
        if predicate is not None:
            return predicate
        if isinstance(bloq, Controlled) and isinstance(
            bloq.subbloq, (QROM, Xor, CSwapApprox)
        ):
            gates = reference_arithmetic_gates(
                bloq, inputs, (), {}, keep_exchange_leaves=True
            )
            if isinstance(bloq.subbloq, QROM):
                loads.append(gates)
            return gates, inputs
        if isinstance(bloq, (QROM, Xor, CSwapApprox)):
            assert all(inputs[name] == wires for name, wires in desired.items())
            if isinstance(bloq, QROM):
                gates = source_qrom_gates(bloq, inputs, work, bits)
                loads.append(gates)
            elif isinstance(bloq, Xor):
                gates = reference_arithmetic_gates(bloq, inputs, (), {})
            else:
                gates = source_relative_swap(
                    inputs["ctrl"][0], inputs["x"], inputs["y"]
                )
            return gates, inputs
        return None

    graph = source
    if adjoint:
        # The source creates adjoint allocations from forward Free nodes,
        # in reverse graph order. Bind their physical names by that graph.
        forward = source.decompose_bloq()
        forward_ports = {reg.name: ports[reg.name] for reg in source.signature}
        walk_source_graph(forward, forward_ports, forward_ports, leaf)
        forward_nodes = [node for node, _, _ in forward.iter_bloqnections()]
        free_nodes = [node for node in forward_nodes if isinstance(node.bloq, Free)]
        freed_by_node = dict(zip(free_nodes, freed, strict=True))
        allocations = {
            index: freed_by_node[node]
            for index, node in enumerate(reversed(forward_nodes))
            if node in freed_by_node
        }
        graph = source.adjoint().decompose_bloq()
        allocate = iter(
            tuple(
                allocations[node.i]
                for node, _, _ in graph.iter_bloqnections()
                if isinstance(node.bloq, Allocate)
            )
        )
        freed.clear()
        loads.clear()
        inverse_graph = True
    if outer_values:
        base = source.adjoint() if adjoint else source
        graph = base.controlled(
            CtrlSpec(cvs=outer_values[0] if len(outer_values) == 1 else outer_values)
        )
    gates, outputs = walk_source_graph(graph, ports, ports, leaf)
    assert outputs == ports and next(allocate, None) is None
    assert Counter(freed) == Counter(words)
    assert len(loads) == 2
    return gates, tuple(loads)


CASES = (
    ((1,), (0,), 1, (1,), ((1,),)),
    ((3,), (0,), 1, (1,), ((1, 0, 1),)),
    ((4,), (1,), 0, (1,), ((0, 1, 1, 0),)),
    ((5,), (1,), 1, (2,), ((3, 1, 1, 3, 2),)),
    ((3, 4), (1, 1), 1, (2,), (tuple(i % 4 for i in range(12)),)),
    ((4,), (1,), 1, (1, 2), ((1, 1, 0, 1), (2, 3, 3, 0))),
)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("shape,logs,control_count,widths,datasets", CASES)
@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.parametrize("adjoint", [False, True])
def test_source_gate_graph_and_bank_channel(
    shape: tuple[int, ...],
    logs: tuple[int, ...],
    control_count: int,
    widths: tuple[int, ...],
    datasets: tuple[tuple[int, ...], ...],
    dirty: bool,
    adjoint: bool,
) -> None:
    pytest.importorskip("qualtran")
    np = pytest.importorskip("numpy")
    from qualtran.bloqs.data_loading.select_swap_qrom import SelectSwapQROM
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    table = QroamCleanTable(shape, logs, datasets)
    source = SelectSwapQROM.build_from_data(
        *(np.array(data).reshape(shape) for data in datasets),
        target_bitsizes=widths,
        num_controls=control_count,
        log_block_sizes=logs,
        use_dirty_ancilla=dirty,
    )
    assert table.batched_data == tuple(
        tuple(map(int, arr.flat)) for arr in source.batched_data
    )
    total = 0

    def alloc(size: int) -> tuple[int, ...]:
        nonlocal total
        start = total
        total += size
        return tuple(range(start, total))

    controls = alloc(control_count)
    selectors = tuple(alloc((extent - 1).bit_length()) for extent in shape)
    targets = tuple(alloc(width) for width in widths)
    bank_wires = alloc(sum(widths) * prod(table.block_shape))
    bank_pool = (
        RegisterWorkspace((), borrowed=bank_wires)
        if dirty
        else RegisterWorkspace(bank_wires)
    )
    leases = tuple(
        tuple(
            bank_pool.allocate(width, dirty=dirty)
            for _ in range(prod(table.block_shape))
        )
        for width in widths
    )
    banks = tuple(tuple(lease.wires for lease in bank) for bank in leases)
    work = alloc(table.workspace_size(control_count, tuple(map(len, selectors))))
    bits = tuple(range(30, 30 + len(work)))
    ports = {
        reg.name: wires
        for reg, wires in zip(
            source.signature,
            (*((controls,) if controls else ()), *filter(None, selectors), *targets),
            strict=True,
        )
    }
    gates = select_swap_qrom_ir(
        table,
        controls,
        selectors,
        targets,
        banks,
        work,
        bits,
        dirty=dirty,
        adjoint=adjoint,
    )
    expected, loads = original_select_swap_gates(
        source, ports, banks, work, bits, adjoint=adjoint
    )
    for bank in reversed(leases):
        for lease in reversed(bank):
            bank_pool.free(lease)
    assert bank_pool.allocate(len(bank_wires), dirty=dirty).wires == bank_wires
    assert_source(gates, expected)
    graph = source.adjoint() if adjoint else source
    model = get_cost_value(graph, QECGatesCost())
    estimate = estimate_component_resources(
        source_select_swap_qrom_cost(
            table, control_count, tuple(map(len, selectors)), widths, dirty=dirty
        )
    )
    assert {k: v for k, v in asdict(estimate.gates).items() if v} == model.asdict()
    assert estimate.peak_logical_qubits == get_cost_value(graph, QubitCount())
    counts = Counter(g.name for g in gates)
    events = counts["measure"]
    assert counts["t"] + counts["tdg"] == model.total_t_count()
    assert events == counts["reset"] == model.measurement
    assert get_cost_value(graph, QubitCount()) == total

    def put(row: list[int], wires: tuple[int, ...], val: int) -> None:
        for i, wire in enumerate(reversed(wires)):
            row[wire] = val >> i & 1

    before, after, loaded = [], [], []
    addresses = tuple(
        product(
            *(
                range(n * b)
                for n, b in zip(table.batch_shape, table.block_shape, strict=True)
            )
        )
    )
    flat_bank = tuple(wire for bank in banks for word in bank for wire in word)
    bank_states = (
        (
            range(1 << len(flat_bank))
            if len(flat_bank) <= 4
            else (
                0,
                1,
                (1 << len(flat_bank)) - 1,
                sum(1 << i for i in range(0, len(flat_bank), 2)),
            )
        )
        if dirty
        else (0,)
    )
    target_states = (0, (1 << sum(widths)) - 1)
    for enable, address, bank_data, destination in product(
        range(1 << control_count), addresses, bank_states, target_states
    ):
        row = [0] * total
        put(row, controls, enable)
        for wires, val in zip(selectors, address, strict=True):
            put(row, wires, val)
        put(row, flat_bank, bank_data)
        put(row, tuple(w for word in targets for w in word), destination)
        before.append(tuple(row))
        high = tuple(a // b for a, b in zip(address, table.block_shape, strict=True))
        loaded_row = row.copy()
        if enable == (1 << control_count) - 1:
            for src, padded, target, bank in zip(
                source.batched_data, source.padded_data, targets, banks, strict=True
            ):
                for word, val in zip(bank, src[high].flat, strict=True):
                    for i, wire in enumerate(reversed(word)):
                        loaded_row[wire] ^= int(val) >> i & 1
                data = int(padded[address])
                for i, wire in enumerate(reversed(target)):
                    row[wire] ^= data >> i & 1
        loaded.append(tuple(loaded_row))
        after.append(tuple(row))
    # Routing and the intervening XOR leave the restored bank values intact
    # before the second load. Certify every And history at both loads.
    assert certified_and_channel_map(loads[0], tuple(before)) == (
        tuple(loaded),
        events // 2,
    )
    assert certified_and_channel_map(loads[1], tuple(loaded)) == (
        tuple(before),
        events // 2,
    )

    def indices(rows: list[tuple[int, ...]]) -> list[int]:
        return [
            2 * sum(bit << (total - i - 1) for i, bit in enumerate(row)) + spectator
            for row in rows
            for spectator in (0, 1)
        ]

    x = torch.arange(1, 2 * len(before) + 1, dtype=torch.float64)
    amps = torch.complex(x, x * (x % 5 - 2))
    amps /= amps.norm()
    start, end = indices(before), indices(after)
    if total <= 13 and events <= 4:
        opposite = select_swap_qrom_ir(
            table,
            controls,
            selectors,
            targets,
            banks,
            work,
            bits,
            dirty=dirty,
            adjoint=not adjoint,
        )
        state = torch.zeros(1 << (total + 1), dtype=torch.complex128)
        expected_state = torch.zeros_like(state)
        state[start] = amps
        expected_state[end] = amps
        for history in product((0, 1), repeat=events):
            branch = forced_branch(state, gates, history)
            torch.testing.assert_close(
                branch, expected_state * 2 ** (-events / 2), atol=2e-12, rtol=0
            )
            if adjoint:
                torch.testing.assert_close(
                    forced_branch(branch, opposite, tuple(1 - bit for bit in history)),
                    state * 2**-events,
                    atol=2e-12,
                    rtol=0,
                )
    else:
        initial = dict(zip(start, amps.tolist(), strict=True))
        result = dict(zip(end, amps.tolist(), strict=True))
        for seed in (1, 41):
            actual = execute_sparse_register(
                CircuitIR(total + 1, gates), initial, seed=seed
            )
            assert all(
                abs(actual.get(k, 0) - result.get(k, 0)) < 2e-12
                for k in actual.keys() | result.keys()
            )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "shape,widths", [((5,), (2,)), ((64,), (1,)), ((10, 32), (1, 2)), ((4,), (3,))]
)
def test_concrete_source_block_minimizer(
    shape: tuple[int, ...], widths: tuple[int, ...]
) -> None:
    pytest.importorskip("qualtran")
    np = pytest.importorskip("numpy")
    from qualtran.bloqs.data_loading.select_swap_qrom import SelectSwapQROM

    for dirty in (False, True):
        source = SelectSwapQROM.build_from_data(
            *(np.zeros(shape, dtype=int) for _ in widths),
            target_bitsizes=widths,
            use_dirty_ancilla=dirty,
        )
        assert (
            optimal_select_swap_log_block_sizes(shape, widths, dirty=dirty)
            == source.log_block_sizes
        )


def test_select_swap_validation() -> None:
    table = QroamCleanTable((1,), (0,), ((1,),))
    args = (table, (), ((),), ((0,),), (((1,),),), (), ())
    with pytest.raises(ValueError, match="boolean"):
        select_swap_qrom_ir(*args, dirty=1)
    with pytest.raises(ValueError, match="boolean"):
        select_swap_qrom_ir(*args, adjoint=1)
    with pytest.raises(ValueError, match="matching destination"):
        select_swap_qrom_ir(*args[:3], (), *args[4:])
    with pytest.raises(ValueError, match="distinct"):
        select_swap_qrom_ir(*args[:3], ((1,),), *args[4:])
    with pytest.raises(ValueError, match="positive widths"):
        optimal_select_swap_log_block_sizes((1,), (0,))


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "shape,logs", [((4,), (2,)), ((3, 4), (0, 1)), ((2, 4), (1, 1))]
)
@pytest.mark.parametrize("dirty", [False, True])
def test_source_omitted_selector_failures(
    shape: tuple[int, ...], logs: tuple[int, ...], dirty: bool
) -> None:
    pytest.importorskip("qualtran")
    np = pytest.importorskip("numpy")
    from qualtran import BloqError
    from qualtran.bloqs.data_loading.select_swap_qrom import SelectSwapQROM

    data = tuple(i % 4 for i in range(prod(shape)))
    source = SelectSwapQROM.build_from_data(
        np.array(data).reshape(shape),
        target_bitsizes=(2,),
        num_controls=1,
        log_block_sizes=logs,
        use_dirty_ancilla=dirty,
    )
    with pytest.raises(BloqError, match="selection"):
        source.decompose_bloq()
    with pytest.raises(ValueError, match="omitted"):
        QroamCleanTable(shape, logs, (data,))


def test_append_borrowed_bank_example() -> None:
    from examples.algorithms.register_select_swap_qrom import main

    main()


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    ("data", "log", "width", "inner"),
    [
        ((1,), 0, 1, 0),
        ((0, 1), 0, 1, 0),
        ((1, 2, 2, 3), 1, 2, 0),
        ((1, 1, 1, 1), 1, 1, 1),
    ],
)
@pytest.mark.parametrize("outer", [(1,), (0,), (1, 0, 1)])
@pytest.mark.parametrize(
    ("dirty", "adjoint"), [(False, False), (False, True), (True, False), (True, True)]
)
def test_select_swap_generic_outer_control(
    data: tuple[int, ...],
    log: int,
    width: int,
    inner: int,
    outer: tuple[int, ...],
    dirty: bool,
    adjoint: bool,
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import CtrlSpec, DecomposeNotImplementedError, DecomposeTypeError
    from qualtran.bloqs.data_loading.select_swap_qrom import SelectSwapQROM
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    cursor = 0

    def alloc(size: int) -> tuple[int, ...]:
        nonlocal cursor
        start = cursor
        cursor += size
        return tuple(range(start, cursor))

    shape = (len(data),)
    table = QroamCleanTable(shape, (log,), (data,))
    outside = alloc(len(outer))
    controls = alloc(inner)
    selection = alloc((len(data) - 1).bit_length())
    target = alloc(width)
    bank = tuple(alloc(width) for _ in range(1 << log))
    data_width = cursor
    work = alloc(table.workspace_size(inner, (len(selection),)))
    assert not work
    predicate = alloc(len(outer) - 1)
    bits = tuple(range(31, 31 + len(predicate)))
    args = (
        (outside,),
        (ControlRegister(1, outer),),
        table,
        controls,
        (selection,),
        (target,),
        (bank,),
        work,
        (),
        predicate,
        bits,
    )
    actual = control_select_swap_qrom_ir(*args, dirty=dirty, adjoint=adjoint)
    opposite = control_select_swap_qrom_ir(*args, dirty=dirty, adjoint=not adjoint)
    base = SelectSwapQROM.build_from_data(
        data,
        target_bitsizes=(width,),
        num_controls=inner,
        log_block_sizes=(log,),
        use_dirty_ancilla=dirty,
    )
    source = (base.adjoint() if adjoint else base).controlled(
        CtrlSpec(cvs=outer[0] if len(outer) == 1 else outer)
    )
    ports = {
        reg.name: wires
        for reg, wires in zip(
            base.signature, (*filter(None, (controls, selection)), target), strict=True
        )
    }
    ports[next(iter(source.signature)).name] = outside
    reference, _ = original_select_swap_gates(
        base,
        ports,
        (bank,),
        work,
        (),
        adjoint=adjoint,
        outer_values=outer,
        predicate_workspace=predicate,
        predicate_bits=bits,
    )
    assert_source_dependencies(actual, reference)
    counts = Counter(gate.name for gate in actual)
    assert counts["t"] + counts["tdg"] == 4 * len(predicate)
    assert counts["measure"] == counts["reset"] == len(predicate)
    assert get_cost_value(source, QubitCount()) == cursor
    if len(data) == 1:
        assert get_cost_value(source, QECGatesCost()).toffoli == 1 + dirty
    else:
        with pytest.raises(
            (DecomposeNotImplementedError, DecomposeTypeError),
            match="And|ArbitraryClifford",
        ):
            get_cost_value(source, QECGatesCost())

    # Cover all public inputs, arbitrary borrowed banks and an entangled spectator.
    bank_width = len(bank) * width
    bank_mask = (1 << bank_width) - 1
    words = torch.arange(1 << data_width)
    if not dirty:
        words = words[(words & bank_mask) == 0]
    activation = int("".join(map(str, outer)), 2)
    address = (words >> (bank_width + width)) & ((1 << len(selection)) - 1)
    enabled = (words >> (data_width - len(outer))) == activation
    if inner:
        enabled &= ((words >> (bank_width + width + len(selection))) & 1) == 1
    selected = torch.tensor(data)[address] << bank_width
    end = torch.where(enabled, words ^ selected, words)
    indices = (words << (len(predicate) + 1)).repeat_interleave(2)
    result_indices = (end << (len(predicate) + 1)).repeat_interleave(2)
    spectator = torch.arange(indices.numel()) & 1
    weights = torch.arange(1, indices.numel() + 1, dtype=torch.float64)
    amplitudes = torch.complex(weights, weights * (weights % 5 - 2))
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (cursor + 1), dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    initial[indices | spectator] = amplitudes
    expected[result_indices | spectator] = amplitudes
    for history in product((0, 1), repeat=len(predicate)):
        result = forced_branch(initial, actual, history)
        torch.testing.assert_close(
            result, expected * 2 ** (-len(predicate) / 2), atol=2e-12, rtol=0
        )
        torch.testing.assert_close(
            forced_branch(result, opposite, tuple(1 - bit for bit in history)),
            initial * 2 ** (-len(predicate)),
            atol=2e-12,
            rtol=0,
        )


@pytest.mark.ftqc_reference
def test_generic_outer_control_rejects_source_non_thru_and() -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.data_loading.select_swap_qrom import SelectSwapQROM
    from qualtran.bloqs.mcmt import And

    source = SelectSwapQROM.build_from_data(
        (1, 2), num_controls=1, log_block_sizes=(0,)
    )
    leaves = {
        str(node.bloq)
        for node in source.controlled().decompose_bloq().flatten().bloq_instances
    }
    assert {"C[And]", "C[And†]"} <= leaves
    for bloq in (And(), And().adjoint()):
        with pytest.raises(DecomposeTypeError, match="non-thru registers"):
            bloq.controlled().decompose_bloq()
    with pytest.raises(ValueError, match="cannot decompose controlled And"):
        control_select_swap_qrom_ir(
            ((0,),),
            (ControlRegister(1, (1,)),),
            QroamCleanTable((2,), (0,), ((1, 2),)),
            (1,),
            ((2,),),
            ((3, 4),),
            (((5, 6),),),
            (7,),
            (30,),
            (),
            (),
        )
