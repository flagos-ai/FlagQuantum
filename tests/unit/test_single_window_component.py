"""Stored Montgomery windows retain source calls and exact complex channels."""

from itertools import product
from math import sqrt

import pytest
import torch

from flagquantum.compiler.registers.modular.single_window_component import (
    single_window_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from tests.unit.measurement_checks import forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_component_hierarchy_control import compare_layers

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("n,w,p", [(1, 1, 1), (3, 1, 3), (4, 2, 5), (5, 5, 13)])
def test_complete_window_root_connections(n: int, w: int, p: int) -> None:
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import SingleWindowModMul

    graph = single_window_program(n, w, p)
    compare_layers(
        graph,
        SingleWindowModMul(w, n, p),
        {f.name for f in graph.functions if f.name != "main"},
    )


@pytest.mark.parametrize("n,w,p", [(1, 1, 1), (3, 1, 3), (4, 2, 5)])
def test_source_complex_window_channel_and_specialized_inverse(
    n: int, w: int, p: int
) -> None:
    np = pytest.importorskip("numpy")
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import SingleWindowModMul

    source = SingleWindowModMul(w, n, p)
    widths = (1,) * w + (n,) + (1,) * (n + w) + (w,)
    ports = []
    logical = 0
    for width in widths:
        ports.append(tuple(range(logical, logical + width)))
        logical += width
    bound = bind_register_program(
        single_window_program(n, w, p),
        tuple(ports),
        clean=tuple(range(logical + 1, logical + 12 * n + 20)),
    )
    total = max(q for gate in bound.circuit.instructions for q in gate.wires) + 1
    gates = CircuitIR(total, bound.circuit.instructions)
    clean = tuple(q for q in range(logical + 1, total))
    inverse = bind_register_program(
        single_window_program(n, w, p, adjoint=True), bound.outputs, clean=clean
    )
    assert inverse.outputs == tuple(ports)
    initial: dict[int, complex] = {}
    expected: dict[int, complex] = {}
    cases = list(product(range(1 << w), range(p), range(2 * p), (0, 1)))
    norm = sqrt(sum(abs(complex(i + 1, 5 - 2 * i)) ** 2 for i in range(len(cases))))

    def encode(words: tuple[tuple[int, ...], ...], values: tuple[int, ...]) -> int:
        return sum(
            ((value >> (len(word) - 1 - i)) & 1) << (total - q - 1)
            for word, value in zip(words, values, strict=True)
            for i, q in enumerate(word)
        )

    for i, (x, y, target, spectator) in enumerate(cases):
        xs = tuple((x >> bit) & 1 for bit in range(w))
        ts = tuple((target >> bit) & 1 for bit in reversed(range(n + w)))
        result = source.on_classical_vals(
            x=np.array(xs), y=y, target=np.array(ts), qrom_index=0
        )
        input_values = (*xs, y, *ts, 0)
        output_values = (
            *tuple(result["x"]),
            int(result["y"]),
            *tuple(result["target"]),
            int(result["qrom_index"]),
        )
        spectator_mask = spectator << (total - logical - 1)
        amplitude = complex(i + 1, 5 - 2 * i) / norm
        initial[encode(tuple(ports), input_values) | spectator_mask] = amplitude
        expected[encode(bound.outputs, output_values) | spectator_mask] = amplitude
    for seed in (0, 7, 23):
        actual = execute_sparse_register(gates, initial, seed=seed)
        assert actual.keys() == expected.keys()
        assert max(abs(actual[k] - expected[k]) for k in expected) < 3e-12
        restored = execute_sparse_register(
            CircuitIR(total, inverse.circuit.instructions), actual, seed=seed
        )
        assert restored.keys() == initial.keys()
        assert max(abs(restored[k] - initial[k]) for k in initial) < 3e-12
    if n == 1:
        events = sum(g.name == "measure" for g in gates.instructions)
        assert events <= 7
        state = torch.zeros(1 << total, dtype=torch.complex128)
        target_state = torch.zeros_like(state)
        for key, value in initial.items():
            state[key] = torch.tensor(value, dtype=torch.complex128)
        for key, value in expected.items():
            target_state[key] = torch.tensor(value, dtype=torch.complex128)
        for history in product((0, 1), repeat=events):
            torch.testing.assert_close(
                forced_branch(state, gates.instructions, history),
                target_state / 2 ** (events / 2),
                rtol=0,
                atol=3e-12,
            )
