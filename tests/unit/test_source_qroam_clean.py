"""Original QROAMClean graph, signed output banks, and measured scratch."""

from collections import Counter
from itertools import product
from math import prod
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.source_qroam_clean import (
    QroamCleanTable,
    source_qroam_clean_ir,
)
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_relative_controlled_swap import source_relative_swap
from tests.unit.test_source_qrom import source_qrom_gates

pytestmark = pytest.mark.unit


def source_clean_graph(
    source: Any,
    ports: SourcePorts,
    outputs: SourcePorts,
    work: tuple[int, ...],
    bits: tuple[int, ...],
) -> tuple[tuple[Instruction, ...], int, list[SourcePorts]]:
    """Bind output allocations by actual QROM and SWZ Soquet edges."""
    import cirq
    import numpy as np
    from qualtran.bloqs.bookkeeping import Allocate, Partition
    from qualtran.bloqs.data_loading.qrom import QROM
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox
    from qualtran.bloqs.swap_network.swap_with_zero import SwapWithZero

    cb = source.decompose_bloq()
    predecessors = {}
    allocations = set()
    for instance, inputs, returned in cb.iter_bloqsoqs():
        bloq = instance.bloq
        if isinstance(bloq, Allocate):
            assert not bloq.dirty
            allocations.add(returned[0])
        if isinstance(bloq, (QROM, SwapWithZero)):
            for reg, soqs in zip(bloq.signature.rights(), returned, strict=True):
                if reg.name.startswith("target"):
                    for before, after in zip(
                        np.asarray(inputs[reg.name], dtype=object).flat,
                        np.asarray(soqs, dtype=object).flat,
                        strict=True,
                    ):
                        predecessors[after] = before

    final = cb.final_soqs()
    bindings = {}
    for reg in source.signature.rights():
        if not reg.name.startswith(("target", "junk_target")):
            continue
        width = int(reg.bitsize)
        for i, soq in enumerate(np.asarray(final[reg.name], dtype=object).flat):
            while soq in predecessors:
                soq = predecessors[soq]
            assert soq in allocations and soq not in bindings
            bindings[soq] = outputs[reg.name][i * width : (i + 1) * width]
    assert set(bindings) == allocations

    qrom_lengths = []
    exchanges = []

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, Allocate):
            assert not inputs and len(desired["reg"]) == int(bloq.dtype.num_qubits)
            return (), desired
        if isinstance(bloq, Partition):
            arrays = {
                reg.name: np.array(
                    [cirq.LineQubit(w) for w in inputs[reg.name]], dtype=object
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
        if isinstance(bloq, QROM):
            gates = source_qrom_gates(bloq, inputs, work, bits)
            qrom_lengths.append(len(gates))
            assert all(inputs[name] == wires for name, wires in desired.items())
            return gates, inputs
        if isinstance(bloq, CSwapApprox):
            exchanges.append(inputs)
            assert all(inputs[name] == wires for name, wires in desired.items())
            return (
                source_relative_swap(inputs["ctrl"][0], inputs["x"], inputs["y"]),
                inputs,
            )
        return None

    gates, returned = walk_source_graph(source, ports, outputs, leaf, bindings=bindings)
    assert returned == outputs and len(qrom_lengths) == 1
    return gates, qrom_lengths[0], exchanges


@pytest.mark.parametrize(
    "shape,logs,control_count,target_widths,datasets,static_extra",
    [
        ((1,), (0,), 0, (1,), ((1,),), -1),
        ((4,), (1,), 0, (1,), ((1, 1, 1, 1),), 0),
        ((4,), (1,), 1, (1,), ((1, 1, 1, 1),), 1),
        ((5,), (1,), 1, (2,), ((3, 1, 1, 3, 2),), 0),
        ((9,), (2,), 0, (1,), ((1, 0, 1, 1, 1, 0, 0, 1, 1),), 0),
        ((3, 4), (1, 1), 0, (2,), (tuple(i % 4 for i in range(12)),), 0),
        ((2, 3), (0, 0), 2, (1,), ((1, 0, 1, 0, 0, 1),), 0),
        ((4,), (1,), 1, (1, 2), ((1, 1, 0, 1), (2, 3, 3, 0)), 0),
    ],
)
def test_original_graph_signed_banks_and_branch_cleanup(
    shape: tuple[int, ...],
    logs: tuple[int, ...],
    control_count: int,
    target_widths: tuple[int, ...],
    datasets: tuple[tuple[int, ...], ...],
    static_extra: int,
) -> None:
    import cirq
    import numpy as np
    from qualtran.bloqs.data_loading.qroam_clean import QROAMClean
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    table = QroamCleanTable(shape, logs, datasets)
    source = QROAMClean.build_from_data(
        *(np.array(data).reshape(shape) for data in datasets),
        target_bitsizes=target_widths,
        num_controls=control_count,
        log_block_sizes=logs,
    )
    assert table.batched_data == tuple(
        tuple(int(v) for v in arr.flat) for arr in source.batched_data
    )
    width = 0

    def allocate(count: int) -> tuple[int, ...]:
        nonlocal width
        start = width
        width += count
        return tuple(range(start, width))

    controls = allocate(control_count)
    selection_widths = tuple((extent - 1).bit_length() for extent in shape)
    selections = tuple(allocate(n) for n in selection_widths)
    banks = tuple(
        tuple(allocate(n) for _ in range(prod(table.block_shape)))
        for n in target_widths
    )
    work = allocate(table.workspace_size(control_count, selection_widths))
    bits = tuple(range(100, 100 + len(work)))
    ports = {"control": controls} if controls else {}
    ports.update(
        zip(
            (reg.name for reg in source.selection_registers),
            filter(None, selections),
            strict=True,
        )
    )
    outputs = dict(ports)
    for i, bank in enumerate(banks):
        outputs[f"target{i}_"] = bank[0]
        if len(bank) > 1:
            outputs[f"junk_target{i}_"] = tuple(w for row in bank[1:] for w in row)

    gates = source_qroam_clean_ir(table, controls, selections, banks, work, bits)
    original, qrom_length, exchanges = source_clean_graph(
        source, ports, outputs, work, bits
    )
    assert_source(gates, original)
    counts = Counter(g.name for g in gates)
    events = counts["measure"]
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count()
    assert events == counts["reset"] == model.measurement
    assert get_cost_value(source, QubitCount()) == width + static_extra

    # Evaluate the original swap leaves as signed basis permutations. Compose
    # them with an independent classical high-block load from the source data.
    matrices = {}
    for n in set(target_widths):
        q = cirq.LineQubit.range(1 + 2 * n)
        matrices[n] = cirq.unitary(cirq.Circuit(cirq.decompose(CSwapApprox(n).on(*q))))

    def put(state: list[int], wires: tuple[int, ...], value: int) -> None:
        for i, wire in enumerate(wires):
            state[wire] = (value >> (len(wires) - i - 1)) & 1

    def read(state: list[int], wires: tuple[int, ...]) -> int:
        return sum(state[wire] << (len(wires) - i - 1) for i, wire in enumerate(wires))

    basis_inputs, loaded, routed, phases = [], [], [], []
    for control, address in product(
        range(1 << control_count),
        product(
            *(
                range(n * block)
                for n, block in zip(table.batch_shape, table.block_shape, strict=True)
            )
        ),
    ):
        initial = [0] * width
        put(initial, controls, control)
        for register, value in zip(selections, address, strict=True):
            put(initial, register, value)
        basis_inputs.append(tuple(initial))
        high = tuple(
            value // block
            for value, block in zip(address, table.block_shape, strict=True)
        )
        enabled = control == (1 << control_count) - 1
        if enabled:
            for source_data, bank in zip(source.batched_data, banks, strict=True):
                for word, value in zip(bank, source_data[high].flat, strict=True):
                    put(initial, word, int(value))
        loaded.append(tuple(initial))
        phase = 1.0
        for exchange in exchanges:
            x, y = exchange["x"], exchange["y"]
            wires = (*exchange["ctrl"], *x, *y)
            column = matrices[len(x)][:, read(initial, wires)]
            destination = int(np.argmax(np.abs(column)))
            assert abs(column[destination]) == pytest.approx(1)
            phase *= float(column[destination].real)
            put(initial, wires, destination)
        if enabled:
            for i, bank in enumerate(banks):
                expected = int(source.padded_data[i][address])
                assert read(initial, bank[0]) == expected
        else:
            assert not any(initial[w] for bank in banks for word in bank for w in word)
        assert not any(initial[w] for w in work)
        routed.append(tuple(initial))
        phases.append(phase)
    assert certified_and_channel_map(gates[:qrom_length], tuple(basis_inputs)) == (
        tuple(loaded),
        events,
    )
    if not control_count and shape == (4,) and logs == (1,):
        assert any(p < 0 for p in phases)

    # Compare complete unnormalized branches with a coherent spectator.
    def indices(rows: list[tuple[int, ...]]) -> list[int]:
        return [
            2 * sum(bit << (width - i - 1) for i, bit in enumerate(row)) + s
            for row in rows
            for s in (0, 1)
        ]

    state = torch.zeros(1 << (width + 1), dtype=torch.complex128)
    expected = torch.zeros_like(state)
    weights = torch.arange(1, 2 * len(basis_inputs) + 1, dtype=torch.float64)
    coefficients = torch.complex(weights, weights * (weights % 5 - 2))
    coefficients /= coefficients.norm()
    state[indices(basis_inputs)] = coefficients
    expected[indices(routed)] = coefficients * torch.tensor(
        phases, dtype=torch.float64
    ).repeat_interleave(2)
    histories = (
        product((0, 1), repeat=events)
        if events <= 4
        else ((0,) * events, (1,) * events)
    )
    for outcomes in histories:
        actual = forced_branch(state, gates, outcomes)
        torch.testing.assert_close(
            actual, expected * 2 ** (-events / 2), atol=2e-12, rtol=0
        )


@pytest.mark.parametrize(
    "shape,logs",
    [((4,), (2,)), ((2, 4), (1, 1)), ((3, 4), (1, 0)), ((3, 4), (0, 1))],
)
def test_locked_source_omitted_selectors(
    shape: tuple[int, ...], logs: tuple[int, ...]
) -> None:
    import numpy as np
    from qualtran.bloqs.data_loading.qroam_clean import QROAMClean

    dataset = tuple(range(prod(shape)))
    source = QROAMClean.build_from_data(
        np.array(dataset).reshape(shape), log_block_sizes=logs
    )
    with pytest.raises(ValueError, match="omitted"):
        QroamCleanTable(shape, logs, (dataset,))
    from qualtran._infra.composite_bloq import BloqError

    with pytest.raises(BloqError):
        source.decompose_bloq()


def test_rejects_bad_table_and_bank_bindings() -> None:
    with pytest.raises(ValueError, match="one block exponent"):
        QroamCleanTable((), (), ((1,),))
    with pytest.raises(ValueError, match="fit"):
        QroamCleanTable((3,), (2,), ((1, 0, 1),))
    with pytest.raises(ValueError, match="matching"):
        QroamCleanTable((3,), (1,), ((1, 0),))
    table = QroamCleanTable((3,), (1,), ((1, 0, 1),))
    with pytest.raises(ValueError, match="selection widths"):
        table.workspace_size(0, (1,))
    with pytest.raises(ValueError, match="one bank word"):
        source_qroam_clean_ir(table, (), ((0, 1),), (((2,),),), (), ())


def test_source_adjoint_wrapper_and_padded_leaf_models() -> None:
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.data_loading.qroam_clean import QROAMClean
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    forward = QROAMClean.build_from_data(
        [3, 1, 1, 3, 2], target_bitsizes=(2,), num_controls=1, log_block_sizes=(1,)
    )
    wrapper = forward.adjoint().with_log_block_sizes((0,))
    leaf = wrapper.qroam_clean_adjoint_bloq
    (instance,) = wrapper.decompose_bloq().bloq_instances
    assert instance.bloq == leaf
    assert leaf.data_shape == (6,)
    assert get_cost_value(wrapper, QECGatesCost()).toffoli == 3
    assert get_cost_value(leaf, QECGatesCost()).toffoli == 4
    with pytest.raises(DecomposeNotImplementedError):
        leaf.decompose_bloq()
