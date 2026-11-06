"""Locked Montgomery composition, wire lifetimes, and full complex output."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.single_window_mod_mul import (
    single_window_qroam_table,
)
from flagquantum.compiler.registers.windowed_mod_mul import windowed_mod_mul_ir
from flagquantum.core.ir import CircuitIR, Instruction
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_single_window_mod_mul import original_window_gates

pytestmark = pytest.mark.unit


def original_gates(
    source: Any,
    x: tuple[int, ...],
    y: tuple[int, ...],
    target: tuple[int, ...],
    indices: tuple[int, ...],
    reduced: int,
    padding: tuple[int, ...],
    bank: tuple[tuple[int, ...], ...],
    work: tuple[int, ...],
    bits: tuple[int, ...],
    output: tuple[int, ...],
    adjoint: bool,
) -> tuple[Instruction, ...]:
    pytest.importorskip("qualtran")
    from qualtran import Adjoint
    from qualtran.bloqs.arithmetic.comparison import LessThanConstant
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.bookkeeping import Allocate, Free
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import SingleWindowModMul

    n, w = len(x), len(padding)
    data = {"x": x, "y": y}
    forward = {**data, "target": output, "qrom_indices": indices, "reduced": (reduced,)}
    freed: list[tuple[int, ...]] = []
    allocations = iter(
        (target[:w],) if adjoint else (target, indices, (reduced,), padding)
    )
    pool = (*work, *(wire for word in bank for wire in word))

    def leaf(
        bloq: Any, ports: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        base = bloq.subbloq if isinstance(bloq, Adjoint) else bloq
        if isinstance(bloq, Allocate):
            word = next(allocations)
            assert len(word) == bloq.signature.get_right("reg").bitsize
            return (), {"reg": word}
        if isinstance(bloq, Free):
            freed.append(ports["reg"])
            return (), {}
        if isinstance(base, SingleWindowModMul):
            acc = ports["target"]
            new = {
                **ports,
                "target": (*acc[w:], *acc[:w]) if adjoint else (*acc[-w:], *acc[:-w]),
            }
            assert all(new[name] == wires for name, wires in desired.items())
            return (
                original_window_gates(
                    source=base,
                    ports=ports,
                    outputs=new,
                    bank=bank,
                    work=work,
                    bits=bits,
                    adjoint=adjoint,
                ),
                new,
            )
        if isinstance(base, LessThanConstant):
            # Its source Cirq inverse returns the same exact XOR comparator.
            if adjoint:
                assert base**-1 == base
            assert all(ports[name] == wires for name, wires in desired.items())
            carry = pool[: n + 1]
            return (
                reference_arithmetic_gates(
                    base, ports, carry, dict(zip(carry[1:], bits[:n], strict=True))
                ),
                ports,
            )
        if isinstance(bloq, XGate):
            return (Instruction("x", ports["q"]),), ports
        if type(bloq).__name__ == "Controlled":
            assert all(ports[name] == wires for name, wires in desired.items())
            carry = pool[n : 2 * n - 1]
            return (
                reference_arithmetic_gates(
                    bloq,
                    ports,
                    pool[: 2 * n - 1],
                    dict(zip(carry, bits[: n - 1], strict=True)),
                ),
                ports,
            )
        return None

    src = source.adjoint() if adjoint else source
    gates, returned = walk_source_graph(
        src, forward if adjoint else data, data if adjoint else forward, leaf
    )
    assert returned == (data if adjoint else forward)
    assert next(allocations, None) is None
    assert Counter(freed) == Counter(
        ((reduced,), padding, indices, target) if adjoint else (target[:w],)
    )
    return gates


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "n,w,p", [(3, 1, 3), (4, 2, 5), (5, 1, 9), (5, 5, 13), (6, 2, 7)]
)
def test_locked_windowed_source_and_complex_roundtrip(n: int, w: int, p: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import (
        DirtyOutOfPlaceMontgomeryModMul,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = DirtyOutOfPlaceMontgomeryModMul(n, w, p)
    table = single_window_qroam_table(n, w, p)
    width = 0

    def allocate(size: int) -> tuple[int, ...]:
        nonlocal width
        start = width
        width += size
        return tuple(range(start, width))

    x, y, target, indices = (allocate(n) for _ in range(4))
    (reduced,) = allocate(1)
    padding = allocate(w)
    bank = tuple(allocate(n + w) for _ in range(table.block_shape[0]))
    work = allocate(n + w - 1)
    bits = tuple(range(40, 40 + 2 * n))
    args = (x, y, target, indices, reduced, p, padding, bank, work, bits)
    gates, output = windowed_mod_mul_ir(*args)
    inverse, restored = windowed_mod_mul_ir(*args, adjoint=True)
    assert restored == target and output == (*target[w:], *padding)
    assert get_cost_value(source, QubitCount()) == width

    for stream, adjoint in ((gates, False), (inverse, True)):
        expected_gates = original_gates(
            source,
            x,
            y,
            target,
            indices,
            reduced,
            padding,
            bank,
            work,
            bits,
            output,
            adjoint,
        )
        assert_source(stream, expected_gates)
        component = source._mod_mul_impl._window.qrom
        erase = get_cost_value(component.adjoint(), QECGatesCost())
        load = get_cost_value(component, QECGatesCost())
        model = get_cost_value(source, QECGatesCost())
        counts = Counter(g.name for g in stream)
        assert counts["t"] + counts["tdg"] == model.total_t_count() + n // w * (
            load.total_t_count() - erase.total_t_count()
        )
        assert (
            counts["measure"]
            == counts["reset"]
            == model.measurement + n // w * (load.measurement - erase.measurement)
        )

    def put(row: list[int], wires: tuple[int, ...], value: int) -> None:
        for i, wire in enumerate(reversed(wires)):
            row[wire] = (value >> i) & 1

    initial, expected = [], []
    forward_graph = source.decompose_bloq()
    reverse_graph = source.adjoint().decompose_bloq()
    for a, b in product(range(p), repeat=2):
        inputs = dict(x=a, y=b)
        observed = forward_graph.on_classical_vals(**inputs)
        assert reverse_graph.on_classical_vals(**observed) == inputs
        if (n, w, p, a, b) == (5, 5, 13, 5, 9):
            assert observed == {
                **inputs,
                "target": 27,
                "qrom_indices": 13,
                "reduced": 1,
            }
            assert source.on_classical_vals(**inputs)["target"] == 1
        if (n, w, p, a, b) == (5, 1, 9, 3, 3):
            assert observed["target"] == 18
            assert source.on_classical_vals(**inputs)["target"] == 0
        row = [0] * width
        put(row, x, a)
        put(row, y, b)
        initial.append(tuple(row))
        put(row, output, int(observed["target"]))
        put(row, indices, int(observed["qrom_indices"]))
        row[reduced] = int(observed["reduced"])
        expected.append(tuple(row))
    events = sum(g.name == "measure" for g in gates)
    if table.log_block_sizes == (0,):
        assert certified_and_channel_map(gates, tuple(initial)) == (
            tuple(expected),
            events,
        )
        assert certified_and_channel_map(inverse, tuple(expected)) == (
            tuple(initial),
            events,
        )

    def addresses(rows: list[tuple[int, ...]]) -> list[int]:
        return [
            2 * sum(bit << (width - i - 1) for i, bit in enumerate(row)) + s
            for row in rows
            for s in (0, 1)
        ]

    values = torch.arange(1, 2 * len(initial) + 1, dtype=torch.float64)
    amplitudes = torch.complex(values, values * (values % 5 - 2))
    amplitudes /= amplitudes.norm()
    data = dict(zip(addresses(initial), amplitudes.tolist(), strict=True))
    result = dict(zip(addresses(expected), amplitudes.tolist(), strict=True))
    for stream, start, end in ((gates, data, result), (inverse, result, data)):
        for seed in (1, 41):
            actual = execute_sparse_register(
                CircuitIR(width + 1, stream), start, seed=seed
            )
            assert all(
                abs(actual.get(k, 0) - end.get(k, 0)) < 2e-12
                for k in actual.keys() | end.keys()
            )


def test_windowed_append_uses_forward_output_aliases_for_uncompute() -> None:
    from examples.algorithms.register_windowed_mod_mul import (
        WindowedMontgomeryMultiplication,
    )
    from flagquantum.dynamic import DynamicCircuit

    circuit = DynamicCircuit(21)
    op = WindowedMontgomeryMultiplication()
    circuit.append(op)
    forward = circuit.to_ir()
    circuit.append(WindowedMontgomeryMultiplication(adjoint=True))
    roundtrip = circuit.to_ir()
    assert roundtrip.instructions[: len(forward.instructions)] == forward.instructions
    assert op.target_wires == (7, 8, 13)
    state = {((1 << 3) | 2) << 15: 0.6j, ((2 << 3) | 1) << 15: 0.8}
    actual = execute_sparse_register(roundtrip, state, seed=3)
    assert all(
        abs(actual.get(k, 0) - state.get(k, 0)) < 2e-12
        for k in actual.keys() | state.keys()
    )


def test_windowed_rejects_inconsistent_register_layout() -> None:
    args = (
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (9, 10, 11),
        12,
        3,
        (13,),
        ((14, 15, 16, 17),),
        (18, 19, 20),
        tuple(range(6)),
    )
    with pytest.raises(ValueError, match="2\\*p"):
        windowed_mod_mul_ir(*args[:5], 5, *args[6:])
    with pytest.raises(ValueError, match="equal-width"):
        windowed_mod_mul_ir(args[0], (), *args[2:])
    with pytest.raises(ValueError, match="distinct"):
        windowed_mod_mul_ir(*args[:-1], (0,) * 6)
