"""Check pinned mixed-coefficient binary-field addition on FlagQuantum IR."""

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import encode_kmx
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
    with pytest.raises(ValueError, match="equal"):
        gf2_add_coefficients_ir((True,), (0, 1))
    with pytest.raises(ValueError, match="disjoint"):
        gf2_add_coefficients_ir((1,), (0,), control=1)
    with pytest.raises(ValueError, match="nonnegative"):
        ClassicalFieldCoefficient(-1)
