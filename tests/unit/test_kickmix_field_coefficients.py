"""Check pinned mixed-coefficient binary-field addition on FlagQuantum IR."""

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseProgram
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.finite_field import (
    ClassicalFieldCoefficient,
    gf2_add_coefficients_ir,
)
from flagquantum.compiler.registers.measurement import measure_x_clear_ir
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_kickmix_field_linear import _block

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("kind", ["quantum", "classical", "constant", "mixed"])
@pytest.mark.parametrize("controlled", [False, True])
def test_field_coefficients_match_pinned_broadcast(kind: str, controlled: bool) -> None:
    km = pytest.importorskip("kickmix")
    assert km.__version__ in {"0.1", "0.1.dev0"}
    width = 5
    builder = km.CircuitBuilder()
    source = builder.create_quantum_register(width, name="source")
    target = builder.create_quantum_register(width, name="target")
    ctrl = builder.create_quantum_register(1, name="control")[0]
    bits = builder.create_classical_register(width, name="offset")
    match kind:
        case "quantum":
            native = source
            coefficients = tuple(range(width))
        case "classical":
            native = bits
            coefficients = tuple(ClassicalFieldCoefficient(i) for i in range(width))
        case "constant":
            native = 0b10101
            coefficients = (True, False, True, False, True)
        case _:
            native = km.array([True, bits[1], source[2], False, source[4]])
            coefficients = (True, ClassicalFieldCoefficient(1), 2, False, 4)
    builder.gf2_iadd(native, target=target, control=ctrl if controlled else True)
    block = "c_gf_iadd" if controlled else "gf_iadd"
    expected = _block(str(builder.finish_circuit()), block).splitlines()
    ir = gf2_add_coefficients_ir(
        coefficients[::-1],
        tuple(reversed(range(width, 2 * width))),
        control=2 * width if controlled else None,
    )
    initialize = tuple(ClassicalXor((), bit).instruction() for bit in range(width))
    lowered = lower_reversible_phase(CircuitIR(2 * width + 1, (*initialize, *ir)))
    assert encode_kmx(lowered).splitlines()[1 + width :] == expected


@pytest.mark.parametrize("controlled", [False, True])
def test_mixed_field_addition_preserves_complex_branch_amplitudes(
    controlled: bool,
) -> None:
    # Wires 0 and 1 are the source and common control. Wires 2, 3, and 4
    # receive the source, the recorded X outcome, and the constant one.
    # The last wire is initially zero and yields both HMR outcomes.
    total = 6
    target = (2, 3, 4)
    ir = gf2_add_coefficients_ir(
        (0, ClassicalFieldCoefficient(0), True),
        target,
        control=1 if controlled else None,
    )
    basis = torch.arange(1 << (total - 1))
    amplitudes = torch.complex((basis + 1).double(), (3 * basis - 5).double())
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << total, dtype=torch.complex128)
    state[::2] = amplitudes
    circuit = DynamicCircuit(total, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(total, (*measure_x_clear_ir(5, 0), *ir)))
    results = run_dynamic(circuit, shots=32, seed=41, strategy="trajectory")
    history = results.classical_bits[:, 0]
    assert set(history.tolist()) == {0, 1}
    enabled = (basis >> 3) & 1 if controlled else torch.ones_like(basis)
    for outcome in (0, 1):
        offset = ((basis >> 4) << 2) | (outcome << 1) | 1
        expected = torch.zeros_like(state)
        expected[::2] = amplitudes[basis ^ (enabled * offset)]
        actual = results.final_states.reshape(32, -1)[history == outcome]
        torch.testing.assert_close(
            actual, expected.expand_as(actual), rtol=0, atol=1e-12
        )


def test_mixed_field_addition_validates_operand_roles() -> None:
    assert gf2_add_coefficients_ir((), ()) == ()
    assert gf2_add_coefficients_ir((False,), (0,)) == ()
    assert gf2_add_coefficients_ir((True,), (0,), control=False) == ()
    assert gf2_add_coefficients_ir((True,), (0,), control=True) == (
        gf2_add_coefficients_ir((True,), (0,))
    )
    with pytest.raises(ValueError, match="equal"):
        gf2_add_coefficients_ir((True,), (0, 1))
    with pytest.raises(ValueError, match="disjoint"):
        gf2_add_coefficients_ir((1,), (0,), control=1)
    with pytest.raises(ValueError, match="nonnegative"):
        ClassicalFieldCoefficient(-1)


def _guarded_gate_stream(
    program: ReversiblePhaseProgram,
) -> list[tuple[str, tuple[int, ...], frozenset[int]]]:
    active: list[int] = []
    gates = []
    for op in program.operations:
        if op.opcode == "push_condition":
            assert op.condition_bit is not None
            active.append(op.condition_bit)
        elif op.opcode == "pop_condition":
            active.pop()
        elif op.opcode != "bit_store0":
            condition = () if op.condition_bit is None else (op.condition_bit,)
            gates.append((op.opcode, op.qubits, frozenset((*active, *condition))))
    return gates


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("guard", [0, 1, 2])
def test_outer_classical_control_matches_locked_effective_gate_stream(
    guard: int,
) -> None:
    km = pytest.importorskip("kickmix")
    assert km.__version__ in {"0.1", "0.1.dev0"}
    builder = km.CircuitBuilder()
    source = builder.create_quantum_register(1, name="source")[0]
    target = builder.create_quantum_register(5, name="target")
    bits = builder.create_classical_register(3, name="offset")
    builder.gf2_iadd(
        km.array([True, bits[0], source, bits[1], bits[0]]),
        target=target,
        control=bits[guard],
    )
    upstream = str(builder.finish_circuit())
    start = upstream.index("PUSH_CONDITION")
    source_stream = [
        line
        for line in upstream[start:].splitlines()
        if not line.startswith("INFO_TAG")
    ]
    assert source_stream[0] == f"PUSH_CONDITION if b{guard}"
    assert source_stream[-1] == "POP_CONDITION"
    coefficients = (
        True,
        ClassicalFieldCoefficient(0),
        0,
        ClassicalFieldCoefficient(1),
        ClassicalFieldCoefficient(0),
    )
    add = gf2_add_coefficients_ir(
        coefficients[::-1], (5, 4, 3, 2, 1), control=ClassicalFieldCoefficient(guard)
    )
    initialize = tuple(ClassicalXor((), bit).instruction() for bit in range(3))
    actual = lower_reversible_phase(CircuitIR(6, (*initialize, *add)))
    assert actual.n_qubits == 6 and actual.n_bits == 3
    header = encode_kmx(actual).splitlines()[:4]
    reference = decode_kmx("\n".join((*header, *source_stream)))
    assert _guarded_gate_stream(actual) == _guarded_gate_stream(reference)
    assert len(_guarded_gate_stream(actual)) == 5


@pytest.mark.parametrize("guard", [0, 1])
def test_classical_guard_preserves_every_measured_branch(guard: int) -> None:
    total = 6
    add = gf2_add_coefficients_ir(
        (0, ClassicalFieldCoefficient(1), True),
        (2, 3, 4),
        control=ClassicalFieldCoefficient(guard),
    )
    basis = torch.arange(1 << (total - 1))
    amplitudes = torch.complex((basis + 1).double(), (3 * basis - 5).double())
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << total, dtype=torch.complex128)
    state[::2] = amplitudes
    circuit = DynamicCircuit(total, inputs=state, dtype=torch.complex128)
    circuit.append(
        FieldOperation(
            total, (*measure_x_clear_ir(5, 0), *measure_x_clear_ir(5, 1), *add)
        )
    )
    results = run_dynamic(circuit, shots=64, seed=41, strategy="trajectory")
    history = results.classical_bits[:, :2]
    assert set(map(tuple, history.tolist())) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    for first, second in ((0, 0), (0, 1), (1, 0), (1, 1)):
        enabled = (first, second)[guard]
        offset = ((basis >> 4) << 2) | (second << 1) | 1
        expected = torch.zeros_like(state)
        expected[::2] = amplitudes[basis ^ (enabled * offset)]
        actual = results.final_states.reshape(64, -1)[
            (history[:, 0] == first) & (history[:, 1] == second)
        ]
        torch.testing.assert_close(
            actual, expected.expand_as(actual), rtol=0, atol=1e-12
        )
