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
from flagquantum.compiler.registers.qroam_resource_model import (
    source_select_swap_qrom_cost,
)
from flagquantum.compiler.registers.select_swap_qrom import (
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
    walk_source_graph,
)
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_relative_controlled_swap import source_relative_swap
from tests.unit.test_source_qrom import source_qrom_gates

pytestmark = pytest.mark.unit


def original_select_swap_gates(
    source: Any,
    ports: SourcePorts,
    banks: tuple[tuple[tuple[int, ...], ...], ...],
    work: tuple[int, ...],
    bits: tuple[int, ...],
) -> tuple[tuple[Instruction, ...], tuple[tuple[Instruction, ...], ...]]:
    pytest.importorskip("qualtran")
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    from qualtran.bloqs.arithmetic import Xor
    from qualtran.bloqs.bookkeeping import Allocate, Free, Partition
    from qualtran.bloqs.data_loading.qrom import QROM
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox

    words = tuple(word for bank in banks for word in bank)
    allocate = iter(words)
    freed: list[tuple[int, ...]] = []
    loads = []

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, Allocate):
            word = next(allocate)
            assert bloq.dirty is source.use_dirty_ancilla
            assert len(word) == bloq.dtype.num_qubits
            return (), {"reg": word}
        if isinstance(bloq, Free):
            # The pinned graph uses the default Free flag for borrowed banks too.
            assert not bloq.dirty
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

    gates, outputs = walk_source_graph(source, ports, ports, leaf)
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
def test_source_gate_graph_and_bank_channel(
    shape: tuple[int, ...],
    logs: tuple[int, ...],
    control_count: int,
    widths: tuple[int, ...],
    datasets: tuple[tuple[int, ...], ...],
    dirty: bool,
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
        table, controls, selectors, targets, banks, work, bits, dirty=dirty
    )
    expected, loads = original_select_swap_gates(source, ports, banks, work, bits)
    for bank in reversed(leases):
        for lease in reversed(bank):
            bank_pool.free(lease)
    assert bank_pool.allocate(len(bank_wires), dirty=dirty).wires == bank_wires
    assert_source(gates, expected)
    model = get_cost_value(source, QECGatesCost())
    estimate = estimate_component_resources(
        source_select_swap_qrom_cost(
            table, control_count, tuple(map(len, selectors)), widths, dirty=dirty
        )
    )
    assert {k: v for k, v in asdict(estimate.gates).items() if v} == model.asdict()
    assert estimate.peak_logical_qubits == get_cost_value(source, QubitCount())
    counts = Counter(g.name for g in gates)
    events = counts["measure"]
    assert counts["t"] + counts["tdg"] == model.total_t_count()
    assert events == counts["reset"] == model.measurement
    assert get_cost_value(source, QubitCount()) == total

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
        state = torch.zeros(1 << (total + 1), dtype=torch.complex128)
        expected_state = torch.zeros_like(state)
        state[start] = amps
        expected_state[end] = amps
        for history in product((0, 1), repeat=events):
            torch.testing.assert_close(
                forced_branch(state, gates, history),
                expected_state * 2 ** (-events / 2),
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
