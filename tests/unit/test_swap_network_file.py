"""Compare complete source swap-network fixtures with the native append bindings."""

from collections import Counter
from dataclasses import replace
from hashlib import sha1
from importlib.metadata import version
from math import prod, sqrt
from pathlib import Path
from typing import Any

import pytest
from sympy import Symbol, symbols

from examples.algorithms.swap_network_catalog import (
    MultiplexedSwapExample,
    RelativeSwapExample,
    SwapWithZeroExample,
    relative_swap_call_graph,
    relative_swap_logical_width,
    relative_swap_signature,
    swap_with_zero_call_graph,
    swap_with_zero_signature,
)
from flagquantum.circuit import Circuit
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import (
    assert_complex_stream,
    assert_source_dependencies,
    source_relative_swap,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def source_network() -> Any:
    import kickmix
    from qualtran.bloqs import swap_network

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(swap_network.__file__).parent
    for name, digest in (
        ("__init__.py", "06b91aef8b6c64356a47b42a651ce89f8053bfbe"),
        ("cswap_approx.py", "362798e240461e6fa6269af8820e9436e5645b4a"),
        ("cswap_approx_test.py", "665e1cb122308cb86d7494728ae85d540275806a"),
        ("multiplexed_cswap.py", "d41e2bb20c122be67e14542411eca4d65be06e9a"),
        ("multiplexed_cswap_test.py", "5d0ff34eb3667cc12acc2402b945b0b04cbf3971"),
        ("swap_network.ipynb", "13d0a58fb58bf2e7d9c2867ce4ad68e07d2a9c22"),
        ("swap_with_zero.py", "756704c796ab999b949fd11903d881734304ac2e"),
        ("swap_with_zero_test.py", "ca49bb77d61d71c68b9ca68dce9aa79997bad7b0"),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    assert tuple(
        name
        for name in vars(swap_network)
        if name in {"CSwapApprox", "MultiplexedCSwap", "SwapWithZero"}
    ) == ("CSwapApprox", "MultiplexedCSwap", "SwapWithZero")
    return swap_network


def assert_signature(native: Signature, original: Any) -> None:
    assert [(r.name, r.bitsize, r.shape_symbolic, r.side.name) for r in native] == [
        (r.name, r.bitsize, r._shape, r.side.name) for r in original.signature
    ]


@pytest.mark.parametrize("width", (None, 1, 2, 3, 4, 5, 10, 32, 64))
def test_original_relative_swap(source_network: Any, width: int | None) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    bits = Symbol("n") if width is None else width
    original = source_network.CSwapApprox(bits)
    assert_signature(relative_swap_signature(bits), original)
    root = relative_swap_call_graph(bits)
    children = {child: count for (_, child), count in root.direct_calls.items()}
    assert children == {
        "T": 4 * bits,
        "ArbitraryClifford(1)": 16 * bits,
        "ArbitraryClifford(2)": 6 * bits - 1,
    }
    shape = original.build_call_graph(None)
    assert {
        (
            "T" if type(child).__name__ == "TGate" else f"ArbitraryClifford({child.n})"
        ): count
        for child, count in shape.items()
    } == children
    assert original.t_complexity().t == 4 * bits
    assert original.t_complexity().clifford == 22 * bits - 1
    assert get_cost_value(original, QubitCount()) == relative_swap_logical_width(bits)
    if width is None:
        return
    example = RelativeSwapExample(width)
    circuit = Circuit(example.qubit_count)
    circuit.append(example)
    reference = source_relative_swap(
        0, tuple(range(1, width + 1)), tuple(range(width + 1, 2 * width + 1))
    )
    gates = circuit.to_ir().instructions
    assert gates == reference
    counts = Counter(g.name for g in gates)
    estimate = get_cost_value(original, QECGatesCost())
    assert counts["t"] + counts["tdg"] == estimate.total_t_count()
    assert counts["cx"] == 6 * width - 1
    assert counts["measure"] == 0
    if width == 4:
        assert_complex_stream(gates, reference, gates, example.qubit_count)


@pytest.mark.parametrize(
    "width,rows,word", ((2, 3, 2), (3, 2, 3), (3, 8, 2), (4, 9, 3), (3, 5, 2))
)
def test_original_multiplexed_swap(
    source_network: Any, width: int, rows: int, word: int
) -> None:
    from qualtran import BQUInt, Register
    from qualtran.resource_counting import SympySymbolAllocator

    original = source_network.MultiplexedCSwap(
        Register("selection", BQUInt(width, rows)), word
    )
    example = MultiplexedSwapExample(width, rows, word)
    assert_signature(example.signature, original)
    assert_source_model(example.resource_model(), original)
    assert_source_model(example.resource_model(), original.adjoint())
    if (width, rows, word) == (3, 5, 2):
        source_calls = original.build_call_graph(SympySymbolAllocator())
        expected = {
            f"CSwap({word})": rows,
            "And(1,0)": rows - 2,
            "And†(1,1)": rows - 2,
            "CNOT": rows - 2,
            "X": 2,
        }
        mapping = {"XGate": "X", "CNOT": "CNOT", "CSwap": f"CSwap({word})"}
        calls = {}
        for child, count in source_calls.items():
            kind = type(child).__name__
            if kind == "And":
                key = f"And{'†' if child.uncompute else ''}({child.cv1},{child.cv2})"
            elif kind in mapping:
                key = mapping[kind]
            else:
                assert kind in {"Split", "Join"}
                continue
            calls[key] = count
        assert (
            calls
            == expected
            == {
                child: count
                for (_, child), count in example.call_graph().direct_calls.items()
            }
        )

    work = tuple(range(example.data_width, example.qubit_count))
    bits = tuple(range(len(work)))
    ports = {
        "selection": tuple(range(width)),
        "targets": tuple(range(width, example.data_width - word)),
        "output": tuple(range(example.data_width - word, example.data_width)),
    }
    states, expected_state = multiplexed_test_states(rows, word, len(work))
    for adjoint, initial, desired in (
        (False, states, expected_state),
        (True, expected_state, states),
    ):
        circuit = Circuit(example.qubit_count)
        circuit.append(replace(example, adjoint=adjoint))
        ir = circuit.to_ir()
        source = original.adjoint() if adjoint else original
        expected_gates = reference_arithmetic_gates(
            source,
            ports,
            work,
            dict(zip(work, bits, strict=True)),
            cirq_style=not adjoint,
            preserve_components=True,
        )
        assert_source_dependencies(ir.instructions, expected_gates)
        counts = Counter(g.name for g in ir.instructions)
        assert (
            counts["t"] + counts["tdg"]
            == source.t_complexity().t
            == 4 * (rows - 2) + 7 * rows * word
        )
        assert counts["measure"] == rows - 2
        for seed in (0, 1, 27):
            check_sparse_state(ir, initial, desired, seed)


def multiplexed_test_states(
    rows: int, word: int, work: int
) -> tuple[dict[int, complex], dict[int, complex]]:
    initial: dict[int, complex] = {}
    desired: dict[int, complex] = {}
    for row in range(rows):
        for occupied in (False, True):
            values = tuple(
                (
                    (i + row) % (1 << word)
                    if occupied
                    else (row + 1) % (1 << word) * (i == row)
                )
                for i in range(rows + 1)
            )
            if occupied and values[row] == values[-1]:
                continue
            swapped = list(values)
            swapped[row], swapped[-1] = swapped[-1], swapped[row]
            for state, words in ((initial, values), (desired, swapped)):
                basis = (
                    row << ((rows + 1) * word)
                    | sum(value << ((rows - i) * word) for i, value in enumerate(words))
                ) << work
                state[basis] = 1j if occupied else 1
    norm = sqrt(len(initial))
    return (
        {index: amplitude / norm for index, amplitude in initial.items()},
        {index: amplitude / norm for index, amplitude in desired.items()},
    )


def check_sparse_state(
    ir: CircuitIR, initial: dict[int, complex], expected: dict[int, complex], seed: int
) -> None:
    actual = execute_sparse_register(ir, initial, seed=seed)
    assert (
        max(
            abs(actual.get(i, 0) - expected.get(i, 0))
            for i in actual.keys() | expected.keys()
        )
        < 3e-12
    )


@pytest.mark.parametrize(
    "widths,word,shape",
    (
        ((3,), 5, (1,)),
        ((2,), 2, (3,)),
        ((2,), 3, (4,)),
        ((3,), 2, (5,)),
        ((4,), 1, (10,)),
        ((3,), 64, (5,)),
        ((8,), 32, (4,)),
        ((3,), 2, (2,)),
        ((2, 2), 2, (3, 4)),
        ((4, 3), 2, (15, 5)),
        ((2, 3, 4), 3, (2, 6, 5)),
        ((3,), 2, (4,)),
        ((2, 1), 2, (3, 2)),
    ),
)
def test_original_swap_with_zero(
    source_network: Any, widths: tuple[int, ...], word: int, shape: tuple[int, ...]
) -> None:
    from qualtran.resource_counting import QubitCount, get_cost_value

    example = SwapWithZeroExample(widths, word, shape)
    original = source_network.SwapWithZero(widths, word, shape)
    assert_signature(example.signature, original)
    assert original.t_complexity().t == 4 * word * (prod(shape) - 1)
    assert original.t_complexity().clifford == (22 * word - 1) * (prod(shape) - 1)
    assert get_cost_value(original, QubitCount()) == example.qubit_count
    ports = {
        r.name: wires
        for r, wires in zip(
            original.selection_registers, example.selections, strict=True
        )
    }
    ports["targets"] = tuple(q for target in example.targets for q in target)

    def leaf(
        bloq: Any, inputs: SourcePorts, outputs: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, source_network.CSwapApprox):
            assert all(inputs[name] == wire for name, wire in outputs.items())
            return (
                source_relative_swap(inputs["ctrl"][0], inputs["x"], inputs["y"]),
                inputs,
            )
        return None

    streams = []
    for adjoint in (False, True):
        circuit = Circuit(example.qubit_count)
        circuit.append(replace(example, adjoint=adjoint))
        gates = circuit.to_ir().instructions
        reference, outputs = walk_source_graph(
            original.adjoint() if adjoint else original, ports, ports, leaf
        )
        assert outputs == ports and gates == reference
        counts = Counter(g.name for g in gates)
        assert counts["t"] + counts["tdg"] == 4 * word * (prod(shape) - 1)
        assert counts["measure"] == 0
        streams.append(circuit.to_ir())
    if example.qubit_count <= 15:
        assert_complex_stream(
            streams[0].instructions,
            streams[0].instructions,
            streams[1].instructions,
            example.qubit_count,
        )


def test_symbolic_swap_with_zero(source_network: Any) -> None:
    p, q, r, b, rows, cols, depth = symbols("p q r b P Q R")
    op = source_network.SwapWithZero((p, q, r), b, (rows, cols, depth))
    assert_signature(swap_with_zero_signature((p, q, r), b, (rows, cols, depth)), op)
    assert swap_with_zero_call_graph(b, (rows, cols, depth)).leaf_counts == {
        "CSwapApprox(b)": rows * cols * depth - 1
    }
    assert op.t_complexity().t == 4 * b * (rows * cols * depth - 1)


@pytest.mark.parametrize("shape,upper", (((50,), 2**20), ((7, 9), 2**10)))
def test_original_classical_swap_with_zero_stage_contract(
    source_network: Any, shape: tuple[int, ...], upper: int
) -> None:
    import numpy as np

    from flagquantum.compiler.registers.lookup.swap_with_zero import (
        swap_with_zero_stages,
    )

    rng = np.random.default_rng(42)
    if len(shape) == 2:
        rng.integers(2**20, size=50)
    data = rng.integers(upper, size=shape)
    widths = tuple(extent.bit_length() for extent in shape)
    op = source_network.SwapWithZero(widths, upper.bit_length(), shape)
    schedule = tuple(swap_with_zero_stages(widths, shape))
    names = tuple(reg.name for reg in op.selection_registers)
    for indices in np.ndindex(shape):
        ports = dict(zip(names, indices, strict=True))
        *returned, result = op.call_classically(targets=data.copy(), **ports)
        assert tuple(returned) == indices
        callback_input = data.copy()
        callback = op.on_classical_vals(targets=callback_input, **ports)
        assert callback["targets"] is callback_input

        words = list(data.flat)
        for axis, bit, left, right in schedule:
            if (indices[axis] >> bit) & 1:
                words[left], words[right] = words[right], words[left]
        np.testing.assert_array_equal(result.flatten(), words)
        np.testing.assert_array_equal(callback_input.flatten(), words)
        assert words[0] == data[indices]
        for axis, bit, left, right in reversed(schedule):
            if (indices[axis] >> bit) & 1:
                words[left], words[right] = words[right], words[left]
        assert words == list(data.flat)
