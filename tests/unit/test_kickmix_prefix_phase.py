"""Verify the locked prefix-phase schedule and borrowed-wire restoration."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)
from flagquantum.compiler.registers.phase.masked_prefix_phase import (
    masked_prefix_phase_ir,
)
from flagquantum.compiler.registers.primitives.measurement import measure_x_clear_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("minimize", [False, True])
def test_four_quantum_phase_matches_locked_gate_order(minimize: bool) -> None:
    gates = masked_prefix_phase_ir(
        (0, 1, 2), (3, 4, 5), clean=(6,), minimize_qubits=minimize, measurement_bit=3
    )
    code = encode_kmx(lower_reversible_phase(CircuitIR(7, gates))).splitlines()[1:]
    assert code[:2] == ["CZ q0 q3", "CCZ q0 q1 q4"]
    if minimize:
        assert code[2:] == ["CCX q0 q1 q3", "CCZ q3 q2 q5"] * 2
    else:
        # The caller supplies q6 in zero, matching the native reset precondition.
        assert code[2:] == [
            "CCX q0 q1 q6",
            "CCZ q6 q2 q5",
            "HMR q6 b3",
            "CZ q0 q1 if b3",
        ]


@pytest.mark.parametrize("measured", [False, True])
def test_four_quantum_phase_preserves_coherence_and_cleans_work(measured: bool) -> None:
    total = 7
    basis = torch.arange(64)
    amplitudes = torch.complex((basis + 1).double(), (2 * basis - 5).double())
    state = torch.zeros(1 << total, dtype=torch.complex128)
    state[::2] = amplitudes / amplitudes.norm()
    gates = masked_prefix_phase_ir(
        (0, 1, 2),
        (3, 4, 5),
        clean=(6,) if measured else (),
        measurement_bit=0 if measured else None,
    )
    circuit = DynamicCircuit(total, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(total, gates))
    result = run_dynamic(circuit, shots=32, seed=31, strategy="trajectory")
    bits = [(basis >> shift) & 1 for shift in reversed(range(6))]
    expected = state.clone()
    expected[::2] *= 1 - 2 * (
        (bits[0] & bits[3])
        ^ (bits[0] & bits[1] & bits[4])
        ^ (bits[0] & bits[1] & bits[2] & bits[5])
    )
    assert result.statistics["branch_count"] == (2 if measured else 1)
    torch.testing.assert_close(
        result.final_states.reshape(32, -1), expected.expand(32, -1), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("clean_count", [0, 1, 2, 3])
def test_mixed_mask_prefix_restores_dirty_data_on_every_branch(
    clean_count: int,
) -> None:
    targets = (0, 1, 2, 3, 4, 5)
    masks = (True, ClassicalFieldCoefficient(0), 6, False, 7, 8)
    dirty_count = 3 - clean_count
    dirty = tuple(range(9, 9 + dirty_count))
    clean = tuple(range(9 + dirty_count, 12))
    total = 13
    basis = torch.arange(1 << (9 + dirty_count))
    amplitudes = torch.complex((basis + 1).double(), (2 * basis - 5).double())
    state = torch.zeros(1 << total, dtype=torch.complex128)
    stride = 1 << (clean_count + 1)
    state[::stride] = amplitudes / amplitudes.norm()
    prefix = measure_x_clear_ir(12, 0)
    gates = masked_prefix_phase_ir(
        targets,
        masks,
        clean=clean,
        dirty=dirty,
        measurement_bit=1 if clean_count else None,
    )
    circuit = DynamicCircuit(total, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(total, (*prefix, *gates)))
    result = run_dynamic(circuit, shots=192, seed=41, strategy="trajectory")
    counts = Counter(
        op.opcode
        for op in lower_reversible_phase(CircuitIR(total, (*prefix, *gates))).operations
    )
    assert counts["ccx"] == (14, 11, 8, 5)[clean_count]
    assert counts["ccz"] == (6, 6, 6, 3)[clean_count]
    assert counts["hmr"] == 1 + clean_count
    assert result.statistics["branch_count"] == 1 << (1 + clean_count)
    full = basis * stride
    bits = [(full >> (total - 1 - q)) & 1 for q in range(9)]
    for classical in (0, 1):
        expected = state.clone()
        active = torch.ones_like(full)
        parity = torch.zeros_like(full)
        for bit, mask in zip(
            bits[:6], (1, classical, bits[6], 0, bits[7], bits[8]), strict=True
        ):
            active = active & bit
            parity = parity ^ (active & mask)
        expected[::stride] *= 1 - 2 * parity
        actual = result.final_states.reshape(192, -1)[
            result.classical_bits[:, 0] == classical
        ]
        torch.testing.assert_close(
            actual, expected.expand_as(actual), atol=1e-12, rtol=0
        )


def test_masked_prefix_uses_common_length_and_validates_carry_ownership() -> None:
    assert masked_prefix_phase_ir((), (True,)) == ()
    assert masked_prefix_phase_ir((0,), ()) == ()
    assert masked_prefix_phase_ir((0, 1, 2, 3), (True,)) == (
        masked_prefix_phase_ir((0,), (True,))
    )
    with pytest.raises(ValueError, match="at least n - 3"):
        masked_prefix_phase_ir((0, 1, 2, 3), (True,) * 4)
    with pytest.raises(ValueError, match="classical result slot"):
        masked_prefix_phase_ir((0, 1, 2, 3), (True,) * 4, clean=(4,))
    with pytest.raises(ValueError, match="separate classical result slot"):
        masked_prefix_phase_ir(
            (0, 1, 2, 3),
            (ClassicalFieldCoefficient(0),) * 4,
            clean=(4,),
            measurement_bit=0,
        )
