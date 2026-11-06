"""Source graph and complete window channels with physical target aliases."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.single_window_mod_mul import (
    single_window_mod_mul_ir,
    single_window_qroam_table,
)
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
from tests.unit.test_source_qroam_clean import source_clean_graph

pytestmark = pytest.mark.unit


def original_window_gates(
    source: Any,
    ports: SourcePorts,
    outputs: SourcePorts,
    bank: tuple[tuple[int, ...], ...],
    work: tuple[int, ...],
    bits: tuple[int, ...],
    adjoint: bool,
) -> tuple[Instruction, ...]:
    from qualtran import Adjoint
    from qualtran.bloqs.arithmetic import Add, Xor
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd
    from qualtran.bloqs.data_loading.qroam_clean import (
        QROAMClean,
        QROAMCleanAdjoint,
        QROAMCleanAdjointWrapper,
    )

    address = {"selection": ports["qrom_index"]}
    bank_wires = tuple(wire for word in bank for wire in word)
    lookup_ports = {**address, "target0_": bank[0]}
    if len(bank) > 1:
        lookup_ports["junk_target0_"] = bank_wires[len(bank[0]) :]
    qrom_work = source.qrom.qrom_bloq.signature.n_qubits()
    assert qrom_work == len(address["selection"]) - source.qrom.log_block_sizes[
        0
    ] + len(bank_wires)
    # The concrete correction datasets are nonuniform and use the full high pool.
    scratch_count = len(address["selection"]) - source.qrom.log_block_sizes[0] - 1
    lookup, qrom_length, exchanges = source_clean_graph(
        source.qrom, address, lookup_ports, work[:scratch_count], bits[:scratch_count]
    )
    uncompute = (
        *(
            gate
            for exchange in reversed(exchanges)
            for gate in source_relative_swap(
                exchange["ctrl"][0], exchange["x"], exchange["y"]
            )
        ),
        *lookup[:qrom_length],
    )

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        base = bloq.subbloq if isinstance(bloq, Adjoint) else bloq
        if isinstance(base, (CAdd, Add, Xor)):
            if isinstance(base, CAdd):
                allocation = (*work, *bank_wires)[: 2 * len(ports["y"])]
            elif isinstance(base, Add):
                allocation = work
            else:
                allocation = ()
            slots = dict(zip(allocation, bits[: len(allocation)], strict=True))
            gates = reference_arithmetic_gates(bloq, inputs, allocation, slots)
            assert all(inputs[name] == wires for name, wires in desired.items())
            return gates, inputs
        if isinstance(bloq, QROAMClean):
            assert bloq == source.qrom and inputs == address
            assert all(lookup_ports[name] == wires for name, wires in desired.items())
            return lookup, lookup_ports
        if isinstance(bloq, QROAMCleanAdjointWrapper):
            assert bloq.qroam_clean == source.qrom and inputs == lookup_ports
            # Read the original bookkeeping before binding the independently
            # composed executable reverse at the model-only leaf.
            return None
        if isinstance(bloq, QROAMCleanAdjoint):
            assert inputs == {**address, "target0_": bank_wires}
            assert all(address[name] == wires for name, wires in desired.items())
            return uncompute, address
        return None

    graph = source.adjoint() if adjoint else source
    gates, returned = walk_source_graph(graph, ports, outputs, leaf)
    assert returned == outputs
    return gates


@pytest.mark.parametrize(
    "n,w,modulus", [(1, 1, 1), (2, 1, 3), (3, 1, 3), (4, 2, 5), (5, 5, 13)]
)
def test_source_window_graph_branches_and_bank_cleanup(
    n: int, w: int, modulus: int
) -> None:
    import numpy as np
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import SingleWindowModMul
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = SingleWindowModMul(w, n, modulus)
    table = single_window_qroam_table(n, w, modulus)
    assert table.datasets == (tuple(int(v) for v in source.qrom.data[0]),)
    assert table.log_block_sizes == source.qrom.log_block_sizes
    width = 0

    def allocate(count: int) -> tuple[int, ...]:
        nonlocal width
        start = width
        width += count
        return tuple(range(start, width))

    x, y, target, index = allocate(w), allocate(n), allocate(n + w), allocate(w)
    bank = tuple(allocate(n + w) for _ in range(table.block_shape[0]))
    work = allocate(n + w - 1)
    bits = tuple(range(50, 50 + 2 * n))
    gates, output = single_window_mod_mul_ir(
        x, y, target, index, modulus, bank, work, bits
    )
    inverse, restored = single_window_mod_mul_ir(
        x, y, output, index, modulus, bank, work, bits, adjoint=True
    )
    assert output == (*target[-w:], *target[:-w]) and restored == target
    ports = {"x": x, "y": y, "target": target, "qrom_index": index}
    outputs = {**ports, "target": output}
    assert_source(
        gates, original_window_gates(source, ports, outputs, bank, work, bits, False)
    )
    assert_source(
        inverse, original_window_gates(source, outputs, ports, bank, work, bits, True)
    )

    model = get_cost_value(source, QECGatesCost())
    erase_model = get_cost_value(source.qrom.adjoint(), QECGatesCost())
    forward_model = get_cost_value(source.qrom, QECGatesCost())
    assert get_cost_value(source, QubitCount()) == width
    for stream in (gates, inverse):
        counts = Counter(g.name for g in stream)
        assert counts["t"] + counts["tdg"] == (
            model.total_t_count()
            - erase_model.total_t_count()
            + forward_model.total_t_count()
        )
        assert (
            counts["measure"]
            == counts["reset"]
            == (model.measurement - erase_model.measurement + forward_model.measurement)
        )

    def put(row: list[int], wires: tuple[int, ...], values: Any) -> None:
        for wire, value in zip(wires, values, strict=True):
            row[wire] = int(value)

    def digits(value: int, size: int) -> tuple[int, ...]:
        return tuple((value >> i) & 1 for i in reversed(range(size)))

    if w < 5:
        cases = tuple(product(range(1 << w), range(modulus), range(2 * modulus)))
    else:
        cases = tuple((i, 1, 0) for i in range(1 << w)) + (
            (31, 12, 25),
            (13, 9, 4),
            (0, 0, 0),
        )
    inputs, expected = [], []
    witness = False
    forward_graph, reverse_graph = (
        source.decompose_bloq(),
        source.adjoint().decompose_bloq(),
    )
    for multiplier, operand, accumulator in cases:
        values = dict(
            x=np.array(digits(multiplier, w)[::-1], dtype=np.uint8),
            y=operand,
            target=np.array(digits(accumulator, n + w), dtype=np.uint8),
            qrom_index=0,
        )
        reference = forward_graph.on_classical_vals(**values)
        reverse_reference = reverse_graph.on_classical_vals(**reference)
        assert all(np.array_equal(values[k], reverse_reference[k]) for k in values)
        if 2 * modulus - 1 < 1 << n:
            high_level = source.on_classical_vals(**values)
            assert all(np.array_equal(reference[k], high_level[k]) for k in values)
        if (n, w, modulus, multiplier, operand, accumulator) == (2, 1, 3, 1, 2, 5):
            assert tuple(reference["target"]) == (0, 0, 1)
            assert tuple(source.on_classical_vals(**values)["target"]) == (1, 0, 1)
            witness = True
        initial = [0] * width
        put(initial, x, values["x"])
        put(initial, y, digits(operand, n))
        put(initial, target, values["target"])
        inputs.append(tuple(initial))
        put(initial, output, reference["target"])
        put(initial, index, digits(int(reference["qrom_index"]), w))
        expected.append(tuple(initial))
    if (n, w, modulus) == (2, 1, 3):
        assert witness

    events = sum(g.name == "measure" for g in gates)
    if table.log_block_sizes == (0,):
        assert certified_and_channel_map(gates, tuple(inputs)) == (
            tuple(expected),
            events,
        )
        assert certified_and_channel_map(inverse, tuple(expected)) == (
            tuple(inputs),
            events,
        )

    def addresses(rows: list[tuple[int, ...]]) -> list[int]:
        return [
            2 * sum(bit << (width - i - 1) for i, bit in enumerate(row)) + s
            for row in rows
            for s in (0, 1)
        ]

    weights = torch.arange(1, 2 * len(inputs) + 1, dtype=torch.float64)
    amplitudes = torch.complex(weights, weights * (weights % 5 - 2))
    amplitudes /= amplitudes.norm()
    in_addresses, out_addresses = addresses(inputs), addresses(expected)
    if n == 1:
        state = torch.zeros(1 << (width + 1), dtype=torch.complex128)
        result = torch.zeros_like(state)
        state[in_addresses] = amplitudes
        result[out_addresses] = amplitudes
        for history in product((0, 1), repeat=events):
            torch.testing.assert_close(
                forced_branch(state, gates, history),
                result * 2 ** (-events / 2),
                atol=2e-12,
                rtol=0,
            )
            torch.testing.assert_close(
                forced_branch(result, inverse, history),
                state * 2 ** (-events / 2),
                atol=2e-12,
                rtol=0,
            )
    else:
        state_sparse = dict(zip(in_addresses, amplitudes.tolist(), strict=True))
        result_sparse = dict(zip(out_addresses, amplitudes.tolist(), strict=True))
        for seed in (0, 31):
            for stream, a, b in (
                (gates, state_sparse, result_sparse),
                (inverse, result_sparse, state_sparse),
            ):
                actual = execute_sparse_register(
                    CircuitIR(width + 1, stream), a, seed=seed
                )
                assert all(
                    abs(actual.get(i, 0) - b.get(i, 0)) < 2e-12
                    for i in actual.keys() | b.keys()
                )


def test_window_rejects_inconsistent_layout() -> None:
    with pytest.raises(ValueError, match="positive divisor"):
        single_window_qroam_table(3, 2, 3)
    with pytest.raises(ValueError, match="positive odd"):
        single_window_qroam_table(3, 1, 2)
    args = ((0,), (1,), (2, 3), (4,), 1, ((5, 6),), (7,), (20, 21))
    with pytest.raises(ValueError, match="index bits"):
        single_window_mod_mul_ir(*args[:3], (), *args[4:])
    with pytest.raises(ValueError, match="bank cell"):
        single_window_mod_mul_ir(*args[:5], (), *args[6:])
    with pytest.raises(ValueError, match="work wires"):
        single_window_mod_mul_ir(*args[:6], (), args[-1])
