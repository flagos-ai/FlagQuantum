"""Literal relative-phase exchange parity against the locked source."""

from collections import Counter
from itertools import product
from math import sqrt

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.primitives.relative_controlled_swap import (
    control_relative_swap_ir,
    relative_controlled_swap_ir,
)
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates

pytestmark = pytest.mark.unit


def source_relative_swap(
    control: int, left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[Instruction, ...]:
    pytest.importorskip("qualtran")
    cirq = pytest.importorskip("cirq")
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox

    op = CSwapApprox(len(left)).on(*map(cirq.LineQubit, (control, *left, *right)))
    names = {
        cirq.CNOT: "cx",
        cirq.H: "h",
        cirq.S: "s",
        cirq.S**-1: "sdg",
        cirq.T: "t",
        cirq.T**-1: "tdg",
    }
    return tuple(
        Instruction(names[gate.gate], tuple(q.x for q in gate.qubits))
        for gate in cirq.decompose(op, keep=lambda gate: gate.gate in names)
    )


def assert_complex_stream(
    actual: tuple[Instruction, ...],
    source: tuple[Instruction, ...],
    inverse: tuple[Instruction, ...],
    width: int,
) -> None:
    pytest.importorskip("qualtran")
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")

    names = {
        "cx": cirq.CNOT,
        "h": cirq.H,
        "s": cirq.S,
        "sdg": cirq.S**-1,
        "t": cirq.T,
        "tdg": cirq.T**-1,
    }
    # Include a spectator and nonfactorized amplitudes over every basis column.
    amplitudes = torch.arange(1 << (width + 1), dtype=torch.float64)
    state = torch.complex(amplitudes + 1, (amplitudes % 7 - 3) * amplitudes)
    state /= state.norm()
    wires = cirq.LineQubit.range(width + 1)
    expected = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(
            cirq.Circuit(
                names[g.name].on(*(wires[q] for q in g.wires)) for g in source
            ),
            qubit_order=wires,
            initial_state=state.numpy().copy(),
        )
        .final_state_vector
    )
    circuit = Circuit(width + 1, inputs=state, dtype=torch.complex128).append(
        FieldOperation(width + 1, actual)
    )
    torch.testing.assert_close(
        circuit.state().flatten(), torch.from_numpy(expected), rtol=0, atol=2e-12
    )
    circuit.append(FieldOperation(width + 1, inverse))
    torch.testing.assert_close(circuit.state().flatten(), state, rtol=0, atol=2e-12)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 2, 3, 5, 8])
def test_original_exchange_stream_and_cost(width: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    left = tuple(range(1, width + 1))
    right = tuple(range(width + 1, 2 * width + 1))
    gates = relative_controlled_swap_ir(0, left, right)
    source = source_relative_swap(0, left, right)
    assert gates == source
    counts = Counter(gate.name for gate in gates)
    assert counts == {
        "t": 2 * width,
        "tdg": 2 * width,
        "h": 8 * width,
        "s": 4 * width,
        "sdg": 4 * width,
        "cx": 6 * width - 1,
    }
    model = get_cost_value(CSwapApprox(width), QECGatesCost())
    assert model.total_t_count() == 4 * width
    assert model.measurement == 0
    assert get_cost_value(CSwapApprox(width), QubitCount()) == 2 * width + 1
    assert_complex_stream(gates, source, gates, 2 * width + 1)


def test_rejects_missing_mismatched_and_aliased_words() -> None:
    for left, right in [((), ()), ((1,), (2, 3)), ((1,), (0,))]:
        with pytest.raises(ValueError):
            relative_controlled_swap_ir(0, left, right)


def assert_source_dependencies(
    actual: tuple[Instruction, ...], expected: tuple[Instruction, ...]
) -> None:
    assert len(actual) == len(expected)
    for wire in {wire for gate in (*actual, *expected) for wire in gate.wires}:
        assert [gate for gate in actual if wire in gate.wires] == [
            gate for gate in expected if wire in gate.wires
        ]


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 2, 3, 5, 8])
def test_literal_adjoint_exchange(width: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    left = tuple(range(1, width + 1))
    right = tuple(range(width + 1, 2 * width + 1))
    source = CSwapApprox(width).adjoint()
    expected = reference_arithmetic_gates(
        source, {"ctrl": (0,), "x": left, "y": right}, (), {}, keep_exchange_leaves=True
    )
    actual = relative_controlled_swap_ir(0, left, right, adjoint=True)
    assert_source_dependencies(actual, expected)
    assert Counter(gate.name for gate in actual) == Counter(
        gate.name for gate in expected
    )
    assert get_cost_value(source, QECGatesCost()).total_t_count() == 4 * width
    assert get_cost_value(source, QubitCount()) == 2 * width + 1
    assert_complex_stream(
        actual, expected, relative_controlled_swap_ir(0, left, right), 2 * width + 1
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("values", [(0,), (1,), (0, 1), (1, 0, 1)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_outer_control_relative_swap(
    width: int, values: tuple[int, ...], adjoint: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import CtrlSpec, DecomposeNotImplementedError
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    size = len(values)
    outside = tuple(range(size))
    left = tuple(range(size + 1, size + width + 1))
    right = tuple(range(size + width + 1, size + 2 * width + 1))
    data_width = size + 2 * width + 1
    workspace = tuple(range(data_width, data_width + size - 1))
    bits = tuple(range(30, 30 + len(workspace)))
    args = (
        (outside,),
        (ControlRegister(1, values),),
        size,
        left,
        right,
        workspace,
        bits,
    )
    actual = control_relative_swap_ir(*args, adjoint=adjoint)
    inverse = control_relative_swap_ir(*args, adjoint=not adjoint)
    base = CSwapApprox(width)
    source = (base.adjoint() if adjoint else base).controlled(
        CtrlSpec(cvs=values[0] if size == 1 else values)
    )
    outer_name = next(iter(source.signature)).name
    expected = reference_arithmetic_gates(
        source,
        {outer_name: outside, "ctrl": (size,), "x": left, "y": right},
        workspace,
        dict(zip(workspace, bits, strict=True)),
        keep_exchange_leaves=True,
    )
    assert_source_dependencies(actual, expected)
    counts = Counter(gate.name for gate in actual)
    assert counts["ccx"] == 6 * width - 1
    assert counts["ch"] == 8 * width
    assert counts["cphase"] == 12 * width
    assert counts["t"] + counts["tdg"] == 4 * len(workspace)
    assert counts["measure"] == len(workspace)
    assert get_cost_value(source, QubitCount()) == data_width + len(workspace)
    with pytest.raises(DecomposeNotImplementedError, match="ArbitraryClifford"):
        get_cost_value(source, QECGatesCost())

    if width > 3:
        # Wider cases exercise the fanout order and resources without dense simulation.
        return
    assert_controlled_exchange_branches(
        actual, expected, inverse, data_width, workspace
    )


def assert_controlled_exchange_branches(
    actual: tuple[Instruction, ...],
    expected: tuple[Instruction, ...],
    inverse: tuple[Instruction, ...],
    data_width: int,
    workspace: tuple[int, ...],
) -> None:
    # Include a spectator entangled across both control sectors and every target input.
    indices = torch.arange(1 << (data_width + 1))
    amplitudes = torch.complex(indices.double() + 1, (indices % 7).double() - 3)
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << (data_width + len(workspace) + 1), dtype=torch.complex128)
    state[((indices >> 1) << (len(workspace) + 1)) | (indices & 1)] = amplitudes
    measurements = sum(gate.name == "measure" for gate in actual)
    for history in product((0, 1), repeat=measurements):
        result = forced_branch(state, actual, history)
        torch.testing.assert_close(
            result, forced_branch(state, expected, history), atol=2e-12, rtol=0
        )
        indices_with_work = torch.arange(state.numel()) >> 1
        assert (
            torch.count_nonzero(
                result[(indices_with_work & ((1 << len(workspace)) - 1)) != 0]
            )
            == 0
        )
        torch.testing.assert_close(
            result.norm(), state.norm() / sqrt(2**measurements), atol=2e-12, rtol=0
        )
        torch.testing.assert_close(
            forced_branch(result, inverse, history),
            state / 2**measurements,
            atol=2e-12,
            rtol=0,
        )
