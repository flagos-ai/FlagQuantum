"""Check Kickmix's equal-width comparison gate flow and coherent cleanup."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.comparison.mixed_comparison import (
    kickmix_greater_than_ir,
    kickmix_less_than_ir,
    kickmix_quantum_compare_ir,
    kickmix_quantum_equal_ir,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)
from flagquantum.compiler.registers.primitives.measurement import measure_x_clear_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_kickmix_field_inverse import _apply_phase_certified_block

pytestmark = pytest.mark.unit


def test_single_bit_clean_carry_matches_pinned_cpp() -> None:
    # Locked gen_flip_if_lt_qq with width one, a quantum control, and one
    # initially clean scratch wire. The source's R q4 is the clean contract.
    ir = kickmix_quantum_compare_ir((0,), (1,), 2, (4,), control=3, measurement_bit=0)
    program = lower_reversible_phase(CircuitIR(5, ir))
    assert encode_kmx(program).splitlines()[1:] == [
        "CX q1 q0",
        "CCX q1 q3 q4",
        "CCX q4 q0 q2",
        "HMR q4 b0",
        "CZ q1 q3 if b0",
        "CX q1 q0",
    ]


def test_mixed_carry_schedule_matches_pinned_cpp() -> None:
    ir = kickmix_quantum_compare_ir((2, 1, 0), (5, 4, 3), 6, or_equal=7)
    program = lower_reversible_phase(CircuitIR(8, ir))
    assert encode_kmx(program).splitlines()[1:] == [
        "X q3",
        "X q4",
        "X q5",
        "X q7",
        "CX q3 q0",
        "CX q3 q7",
        "CCX q7 q0 q3",
        "CX q4 q1",
        "CX q4 q3",
        "CCX q1 q3 q4",
        "CX q5 q2",
        "CX q5 q4",
        "CCX q2 q4 q6",
        "CX q5 q6",
        "CX q5 q2",
        "CX q5 q4",
        "CCX q1 q3 q4",
        "CX q4 q3",
        "CX q4 q1",
        "CCX q7 q0 q3",
        "CX q3 q7",
        "CX q3 q0",
        "X q7",
        "X q3",
        "X q4",
        "X q5",
        "X q6",
    ]


@pytest.mark.parametrize(
    "workspace_width,minimize", [(0, False), (2, False), (4, False), (4, True)]
)
def test_all_basis_inputs_and_measurement_phases(
    workspace_width: int, minimize: bool
) -> None:
    np = pytest.importorskip("numpy")
    width = 4
    left, right = tuple(range(width)), tuple(range(width, 2 * width))
    target, equal, control = 2 * width, 2 * width + 1, 2 * width + 2
    workspace = tuple(range(control + 1, control + 1 + workspace_width))
    inputs = np.arange(1 << (control + 1))
    wires = [(inputs >> bit) & 1 for bit in reversed(range(control + 1))]
    wires.extend(np.zeros_like(inputs) for _ in workspace)
    expected = np.array(wires)
    lhs = inputs >> (width + 3)
    rhs = (inputs >> 3) & ((1 << width) - 1)
    expected[target] ^= (inputs & 1) & (
        (lhs < rhs) | ((lhs == rhs) & ((inputs >> 1) & 1))
    )
    ir = kickmix_quantum_compare_ir(
        left,
        right,
        target,
        workspace,
        or_equal=equal,
        control=control,
        measurement_bit=0,
        minimize_qubits=minimize,
    )
    _apply_phase_certified_block(np, wires, ir, None)
    np.testing.assert_array_equal(wires, expected)
    program = lower_reversible_phase(CircuitIR(len(wires), ir))
    counts = Counter(op.opcode for op in program.operations)
    measured = 0 if minimize else max(0, min(width - 1, workspace_width - 1))
    assert counts["hmr"] == measured
    assert counts["z"] == counts["cz"] == measured
    assert program.n_bits == int(measured > 0)


@pytest.mark.parametrize("clean_width", [0, 1])
def test_narrow_comparison_preserves_complex_amplitudes(clean_width: int) -> None:
    # Wire four is an entangled dirty helper when it is not supplied clean.
    total = 5
    basis = torch.arange(32)
    state = torch.complex((basis + 1).double(), (3 * basis - 7).double())
    if clean_width:
        state[1::2] = 0
    state /= state.norm()
    delta = (((basis >> 4) & 1) < ((basis >> 3) & 1)) & (((basis >> 1) & 1) == 1)
    expected = state[basis ^ (delta.long() << 2)]
    ir = kickmix_quantum_compare_ir(
        (0,), (1,), 2, (4,) if clean_width else (), control=3, measurement_bit=0
    )
    circuit = DynamicCircuit(total, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(total, ir))
    result = run_dynamic(circuit, shots=32, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(32, -1), expected.expand(32, -1), rtol=0, atol=1e-12
    )


def test_classical_equality_and_feedback_preserve_each_complex_branch() -> None:
    total = 10
    basis = torch.arange(1 << 6)
    amplitudes = torch.complex((basis + 1).double(), (3 * basis - 5).double())
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << total, dtype=torch.complex128)
    state[::16] = amplitudes
    circuit = DynamicCircuit(total, inputs=state, dtype=torch.complex128)
    ir = kickmix_quantum_compare_ir(
        (0, 1),
        (2, 3),
        4,
        (6, 7, 8),
        or_equal=ClassicalFieldCoefficient(0),
        control=5,
        measurement_bit=1,
    )
    circuit.append(FieldOperation(total, (*measure_x_clear_ir(9, 0), *ir)))
    result = run_dynamic(circuit, shots=64, seed=41, strategy="trajectory")
    history = result.classical_bits[:, :2]
    assert set(map(tuple, history.tolist())) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    for equality in (0, 1):
        delta = ((basis >> 4) < (((basis >> 2) & 3) + equality)) & ((basis & 1) == 1)
        expected = torch.zeros_like(state)
        expected[::16] = amplitudes[basis ^ (delta.long() << 1)]
        branch = result.final_states.reshape(64, -1)[history[:, 0] == equality]
        torch.testing.assert_close(
            branch, expected.expand_as(branch), rtol=0, atol=1e-12
        )


def test_comparator_validates_workspace_and_preserves_empty_behavior() -> None:
    assert not kickmix_quantum_compare_ir((), (), 0)
    assert kickmix_quantum_compare_ir((), (), 0, or_equal=True)[0].name == "x"
    with pytest.raises(ValueError, match="equal word"):
        kickmix_quantum_compare_ir((), (0,), 1)
    with pytest.raises(ValueError, match="clean carry"):
        kickmix_quantum_compare_ir((0, 1), (2, 3), 4)
    with pytest.raises(ValueError, match="result slot"):
        kickmix_quantum_compare_ir((0,), (1,), 2, (4,), control=3)


@pytest.mark.parametrize("equality", [True, ClassicalFieldCoefficient(0)])
def test_empty_phase_compare_matches_locked_scalar_opcode(
    equality: bool | ClassicalFieldCoefficient,
) -> None:
    from flagquantum.core.classical_dataflow import ClassicalXor

    prefix = (
        (ClassicalXor((), 0).instruction(),)
        if isinstance(equality, ClassicalFieldCoefficient)
        else ()
    )
    ir = (*prefix, *kickmix_quantum_compare_ir((), (), None, or_equal=equality))
    encoded = encode_kmx(lower_reversible_phase(CircuitIR(1, ir)))
    assert encoded.splitlines()[-1] == ("NEG" if equality is True else "NEG if b0")


def test_single_bit_quantum_phase_matches_locked_cpp() -> None:
    ir = kickmix_quantum_compare_ir((0,), (1,), None, (4,), control=3)
    program = lower_reversible_phase(CircuitIR(5, ir))
    assert encode_kmx(program).splitlines()[1:] == [
        "CX q1 q0",
        "CCZ q1 q3 q0",
        "CX q1 q0",
    ]


@pytest.mark.parametrize("clean_width", [0, 2, 3])
def test_phase_comparator_preserves_all_complex_measurement_branches(
    clean_width: int,
) -> None:
    # Quantum equality, quantum enable, and an arbitrary spectator.
    logical = 7
    total = logical + clean_width
    basis = torch.arange(1 << logical)
    amplitudes = torch.complex((basis + 1).double(), (3 * basis - 5).double())
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << total, dtype=torch.complex128)
    stride = 1 << clean_width
    state[::stride] = amplitudes
    lhs, rhs = basis >> 5, (basis >> 3) & 3
    equal, control = (basis >> 2) & 1, (basis >> 1) & 1
    phase = (lhs < rhs + equal) & (control == 1)
    expected = torch.zeros_like(state)
    expected[::stride] = amplitudes * (1 - 2 * phase.long())
    ir = kickmix_quantum_compare_ir(
        (0, 1),
        (2, 3),
        None,
        tuple(range(logical, total)),
        or_equal=4,
        control=5,
        measurement_bit=0,
    )
    circuit = DynamicCircuit(total, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(total, ir))
    result = run_dynamic(circuit, shots=32, seed=41, strategy="trajectory")
    if clean_width:
        assert set(result.classical_bits[:, 0].tolist()) == {0, 1}
    torch.testing.assert_close(
        result.final_states.reshape(32, -1), expected.expand(32, -1), rtol=0, atol=1e-12
    )


def test_empty_phase_compare_preserves_conditional_scalar_in_dynamic_execution() -> (
    None
):
    state = torch.tensor([1 + 2j, 0, 3 - 4j, 0], dtype=torch.complex128)
    state /= state.norm()
    circuit = DynamicCircuit(2, inputs=state, dtype=torch.complex128)
    phase = kickmix_quantum_compare_ir(
        (), (), None, or_equal=ClassicalFieldCoefficient(0)
    )
    circuit.append(FieldOperation(2, (*measure_x_clear_ir(1, 0), *phase)))
    result = run_dynamic(circuit, shots=32, seed=41, strategy="trajectory")
    outcomes = result.classical_bits[:, 0]
    assert set(outcomes.tolist()) == {0, 1}
    expected = (1 - 2 * outcomes.long())[:, None] * state[None, :]
    torch.testing.assert_close(
        result.final_states.reshape(32, -1), expected, rtol=0, atol=1e-12
    )


def test_mixed_equality_matches_locked_cpp_gate_stream() -> None:
    from flagquantum.core.classical_dataflow import ClassicalXor

    ir = kickmix_quantum_equal_ir(
        (1, 0), (ClassicalFieldCoefficient(1), 2), 4, (6, 7), measurement_bit=2
    )
    prefix = (ClassicalXor((), 1).instruction(),)
    program = lower_reversible_phase(CircuitIR(8, (*prefix, *ir)))
    assert encode_kmx(program).splitlines()[2:] == [
        "CX q2 q0",
        "X q1 if b1",
        "X q0",
        "X q1",
        "CX q0 q6",
        "CCX q6 q1 q7",
        "CCX q7 q1 q4",
        "HMR q7 b2",
        "CZ q6 q1 if b2",
        "HMR q6 b2",
        "Z q0 if b2",
        "X q0",
        "X q1",
        "CX q2 q0",
        "X q1 if b1",
    ]


@pytest.mark.parametrize("phase_target", [False, True])
def test_mixed_equality_restores_each_complex_measurement_branch(
    phase_target: bool,
) -> None:
    total = 10
    basis = torch.arange(64)
    amplitudes = torch.complex((basis + 1).double(), (3 * basis - 5).double())
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << total, dtype=torch.complex128)
    state[::16] = amplitudes
    circuit = DynamicCircuit(total, inputs=state, dtype=torch.complex128)
    ir = kickmix_quantum_equal_ir(
        (0, 1, 2),
        (3, ClassicalFieldCoefficient(0), True),
        None if phase_target else 5,
        (6, 7, 8),
        control=4,
        measurement_bit=1,
    )
    circuit.append(FieldOperation(total, (*measure_x_clear_ir(9, 0), *ir)))
    result = run_dynamic(circuit, shots=64, seed=41, strategy="trajectory")
    history = result.classical_bits[:, :2]
    assert set(map(tuple, history.tolist())) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    for coefficient in (0, 1):
        right = (((basis >> 2) & 1) << 2) | (coefficient << 1) | 1
        enabled = ((basis >> 3) == right) & (((basis >> 1) & 1) == 1)
        expected = torch.zeros_like(state)
        if phase_target:
            expected[::16] = amplitudes * (1 - 2 * enabled.long())
        else:
            expected[::16] = amplitudes[basis ^ enabled.long()]
        actual = result.final_states.reshape(64, -1)[history[:, 0] == coefficient]
        torch.testing.assert_close(
            actual, expected.expand_as(actual), rtol=0, atol=1e-12
        )


def test_equality_validates_result_slot_and_workspace() -> None:
    assert kickmix_quantum_equal_ir((), (), 0)[0].name == "x"
    assert not kickmix_quantum_equal_ir((), (), 0, control=False)
    with pytest.raises(ValueError, match="one clean wire"):
        kickmix_quantum_equal_ir((0, 1), (True, False), 2, (3,))
    with pytest.raises(ValueError, match="separate"):
        kickmix_quantum_equal_ir(
            (0, 1), (True, ClassicalFieldCoefficient(0)), 2, (3, 4), measurement_bit=0
        )


@pytest.mark.parametrize("phase_target", [False, True])
def test_full_mixed_less_than_preserves_every_complex_branch(
    phase_target: bool,
) -> None:
    total = 10
    basis = torch.arange(64)
    amplitudes = torch.complex((basis + 1).double(), (3 * basis - 5).double())
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << total, dtype=torch.complex128)
    state[::16] = amplitudes
    circuit = DynamicCircuit(total, inputs=state, dtype=torch.complex128)
    ir = kickmix_less_than_ir(
        (0, 1, 2),
        (3, ClassicalFieldCoefficient(0), True),
        None if phase_target else 5,
        (6, 7, 8),
        or_equal=True,
        control=4,
        measurement_bit=1,
    )
    circuit.append(FieldOperation(total, (*measure_x_clear_ir(9, 0), *ir)))
    result = run_dynamic(circuit, shots=64, seed=41, strategy="trajectory")
    history = result.classical_bits[:, :2]
    assert set(map(tuple, history.tolist())) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    for coefficient in (0, 1):
        right = (((basis >> 2) & 1) << 2) | (coefficient << 1) | 1
        enabled = ((basis >> 3) <= right) & (((basis >> 1) & 1) == 1)
        expected = torch.zeros_like(state)
        if phase_target:
            expected[::16] = amplitudes * (1 - 2 * enabled.long())
        else:
            expected[::16] = amplitudes[basis ^ enabled.long()]
        actual = result.final_states.reshape(64, -1)[history[:, 0] == coefficient]
        torch.testing.assert_close(
            actual, expected.expand_as(actual), rtol=0, atol=1e-12
        )


def test_full_quantum_ladder_preserves_every_independent_outcome_phase() -> None:
    np = pytest.importorskip("numpy")
    width = 4
    left, right = tuple(range(width)), tuple(range(width, 2 * width))
    target, equal, control = 8, 9, 10
    workspace = (11, 12, 13, 14)
    values = np.arange(1 << 11)
    wires = [(values >> bit) & 1 for bit in reversed(range(11))]
    wires.extend(np.zeros_like(values) for _ in workspace)
    expected = np.array(wires)
    lhs = values >> 7
    rhs = (values >> 3) & 15
    expected[target] ^= (values & 1) & (lhs < rhs + ((values >> 1) & 1))
    ir = kickmix_less_than_ir(
        left,
        right,
        target,
        workspace,
        or_equal=equal,
        control=control,
        measurement_bit=0,
    )
    _apply_phase_certified_block(np, wires, ir, None)
    np.testing.assert_array_equal(wires, expected)
    program = lower_reversible_phase(CircuitIR(len(wires), ir))
    assert Counter(op.opcode for op in program.operations)["hmr"] == 4
    assert program.n_bits == 1


def test_exact_less_than_dispatch_and_missing_resources() -> None:
    left, right = (0, 1), (2, 3)
    assert kickmix_less_than_ir(left, right, 4, (5,)) == (
        kickmix_quantum_compare_ir(left, right, 4, (5,))
    )
    with pytest.raises(ValueError, match="Low-space classical"):
        kickmix_less_than_ir(left, (True, False), 4)
    with pytest.raises(ValueError, match="minus one"):
        kickmix_less_than_ir(left, (True, 2), 4)
    with pytest.raises(ValueError, match="result slot"):
        kickmix_less_than_ir(left, (True, 2), 4, (5,))
    with pytest.raises(ValueError, match="separate"):
        kickmix_less_than_ir(
            left, (ClassicalFieldCoefficient(1), 2), 4, (5,), measurement_bit=1
        )


def test_fractional_budget_preserves_locked_unmarked_xzxz_stream() -> None:
    ir = kickmix_less_than_ir(
        (1, 0), (True, False), None, (6,), or_equal=True, btol=1.01
    )
    program = lower_reversible_phase(CircuitIR(7, ir))
    assert encode_kmx(program).splitlines()[1:] == [
        "X q1",
        "Z q1",
        "X q1",
        "Z q1",
        "X q1",
        "X q1",
    ]


def test_bit_tolerance_retains_source_exact_dispatch_threshold() -> None:
    args = ((0, 1), (2, 3), 4, (5,))
    dedicated = kickmix_less_than_ir(*args, measurement_bit=0)
    full = kickmix_less_than_ir(*args, measurement_bit=0, btol=2.0)
    assert "measure" not in {gate.name for gate in dedicated}
    assert "measure" in {gate.name for gate in full}
    assert not kickmix_less_than_ir(*args, btol=1)
    assert not kickmix_less_than_ir((), (), 0, btol=-1)
    assert kickmix_less_than_ir((), (), 0, or_equal=True, btol=0)[0].name == "x"
    with pytest.raises(ValueError, match="ordered"):
        kickmix_less_than_ir(*args, btol=float("nan"))


@pytest.mark.parametrize("budget,retained", [(1.01, 1), (2.0, 1), (2.01, 2)])
def test_fractional_tolerance_preserves_complex_prefix_comparison(
    budget: float,
    retained: int,
) -> None:
    width = 4
    logical = 2 * width + 1
    total = logical + 1
    basis = torch.arange(1 << logical)
    amplitudes = torch.complex((basis + 1).double(), (3 * basis - 5).double())
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << total, dtype=torch.complex128)
    state[::2] = amplitudes
    left, right, target = tuple(range(width)), tuple(range(width, 2 * width)), 8
    ir = kickmix_less_than_ir(left, right, target, (9,), btol=budget, measurement_bit=0)
    assert not {wire for op in ir for wire in op.wires} & set(
        (*left[retained:], *right[retained:])
    )
    circuit = DynamicCircuit(total, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(total, ir))
    result = run_dynamic(circuit, shots=16, seed=41, strategy="trajectory")
    lhs = basis >> (2 * width + 1 - retained)
    rhs = (basis >> (width + 1 - retained)) & ((1 << retained) - 1)
    expected = torch.zeros_like(state)
    expected[::2] = amplitudes[basis ^ (lhs < rhs).long()]
    torch.testing.assert_close(
        result.final_states.reshape(16, -1), expected.expand(16, -1), rtol=0, atol=1e-12
    )


def test_greater_than_classical_equality_matches_pinned_cpp() -> None:
    from flagquantum.core.classical_dataflow import ClassicalXor

    gates = kickmix_greater_than_ir(
        (0,),
        (ClassicalFieldCoefficient(0),),
        None,
        (4,),
        or_equal=ClassicalFieldCoefficient(1),
        control=3,
    )
    initialize = tuple(ClassicalXor((), slot).instruction() for slot in range(2))
    assert encode_kmx(
        lower_reversible_phase(CircuitIR(5, (*initialize, *gates)))
    ).splitlines()[3:] == [
        "BIT_INVERT b1",
        "X q0 if b0",
        "CZ q3 q0 if b0",
        "X q0",
        "CZ q3 q0 if b1",
        "X q0",
        "X q0 if b0",
        "Z q3",
        "BIT_INVERT b1",
    ]


@pytest.mark.parametrize("phase_target", [False, True])
def test_greater_than_restores_classical_equality_and_all_complex_branches(
    phase_target: bool,
) -> None:
    total = 8
    basis = torch.arange(32)
    amplitudes = torch.complex((basis + 1).double(), (3 * basis - 5).double())
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << total, dtype=torch.complex128)
    state[::8] = amplitudes
    gates = kickmix_greater_than_ir(
        (0, 1),
        (2, True),
        None if phase_target else 4,
        (5, 6),
        or_equal=ClassicalFieldCoefficient(0),
        control=3,
        measurement_bit=1,
    )
    prefix = measure_x_clear_ir(7, 0)
    circuit = DynamicCircuit(total, inputs=state, dtype=torch.complex128).append(
        FieldOperation(total, (*prefix, *gates))
    )
    result = run_dynamic(circuit, shots=128, seed=41, strategy="trajectory")
    program = lower_reversible_phase(CircuitIR(total, (*prefix, *gates)))
    counts = Counter(op.opcode for op in program.operations)
    assert counts["bit_invert"] == 2
    assert counts["hmr"] == (2 if phase_target else 3)
    assert result.statistics["branch_count"] == 2 ** counts["hmr"]
    assert set(result.classical_bits[:, 0].tolist()) == {0, 1}
    for equal in (0, 1):
        left = basis >> 3
        right = (((basis >> 2) & 1) << 1) | 1
        enabled = ((left > right) | ((left == right) & bool(equal))) & (
            ((basis >> 1) & 1) == 1
        )
        expected = torch.zeros_like(state)
        if phase_target:
            expected[::8] = amplitudes * (1 - 2 * enabled.long())
        else:
            expected[::8] = amplitudes[basis ^ enabled.long()]
        actual = result.final_states.reshape(128, -1)[
            result.classical_bits[:, 0] == equal
        ]
        torch.testing.assert_close(
            actual, expected.expand_as(actual), rtol=0, atol=1e-12
        )


def test_greater_than_rejects_mutated_equality_aliases_and_keeps_tolerance() -> None:
    coefficient = ClassicalFieldCoefficient(0)
    with pytest.raises(ValueError, match="separate classical slot"):
        kickmix_greater_than_ir((0,), (coefficient,), 1, or_equal=coefficient)
    with pytest.raises(ValueError, match="separate quantum wire"):
        kickmix_greater_than_ir((0,), (1,), 2, or_equal=1)
    assert not kickmix_greater_than_ir((0, 1), (2, 3), 4, control=False)
    gates = kickmix_greater_than_ir((0, 1), (2, 3), 4, or_equal=5, btol=1)
    assert [(op.name, op.wires) for op in gates] == [
        ("x", (5,)),
        ("x", (4,)),
        ("x", (5,)),
    ]
