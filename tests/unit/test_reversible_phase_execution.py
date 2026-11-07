"""Check local execution of the locked guarded measurement effects."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import decode_kmx
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.core.classical_control import validate_classical_control
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.lowering import circuit_to_program
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import ValidationError
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.density_matrix import density_matrix_from_ir

pytestmark = pytest.mark.unit

_SOURCE = (
    'INFO_TAG "@flagquantum:reversible-phase-v1 wires=4 bits=1"\n'
    "HMR q0 b0\n"
    "PUSH_CONDITION if b0\n"
    "HMR q1 b0\n"
    "R q2\n"
    "X q3\n"
    "PUSH_CONDITION if b0\n"
    "X q0\n"
    "POP_CONDITION\n"
    "POP_CONDITION\n"
)


def _binding() -> tuple[torch.Tensor, DynamicCircuit]:
    state = torch.zeros(16, dtype=torch.complex128)
    state[2], state[6] = 1, 2j
    state /= state.norm()
    program = decode_kmx(_SOURCE)
    gates = bind_reversible_phase_execution(program, classical_offset=2)
    circuit = DynamicCircuit(4, inputs=state, dtype=torch.complex128).append(
        FieldOperation(4, gates)
    )
    assert Counter(op.opcode for op in program.operations)["hmr"] == 2
    return state, circuit


@pytest.mark.parametrize("strategy", ["trajectory", "batched"])
def test_guarded_effects_restore_inactive_complex_states(strategy: str) -> None:
    state, circuit = _binding()
    result = run_dynamic(circuit, shots=128, seed=41, strategy=strategy)
    assert result.statistics["branch_count"] == 5
    result_bit, snapshot, nested = result.classical_bits[:, 2:].T
    inactive = snapshot == 0
    active = ~inactive
    final = result.final_states.reshape(128, -1)
    torch.testing.assert_close(final[inactive], state.expand(int(inactive.sum()), -1))
    assert torch.all(result_bit[inactive] == 0)
    assert result.statistics["measurement_count"] == 128 + int(active.sum())
    assert result.statistics["reset_count"] == int(active.sum())
    assert torch.equal(nested, result_bit)
    for outcome in (0, 1):
        actual = final[active & (result_bit == outcome)]
        sign = 1 if outcome == 0 else -1
        amplitude = (1 + sign * 2j) / 5**0.5
        basis_index = 8 * outcome + 1
        expected = torch.zeros_like(state)
        expected[basis_index] = amplitude
        relative = (actual[:, basis_index] / amplitude).real
        assert set(relative.sign().tolist()) == {-1, 1}
        torch.testing.assert_close(
            actual, relative.sign()[:, None] * expected, atol=1e-12, rtol=0
        )


def test_guarded_effects_density_and_unsupported_program_level() -> None:
    state, circuit = _binding()
    conversion = circuit_to_program(circuit.to_ir())
    assert any(
        "conditional effect result merge" in diagnostic.message
        for diagnostic in conversion.diagnostics
    )
    rho = density_matrix_from_ir(circuit).reshape(16, 16)
    expected = 0.5 * torch.outer(state, state.conj())
    expected[1, 1] += 0.25
    expected[9, 9] += 0.25
    torch.testing.assert_close(rho, expected, atol=1e-12, rtol=0)


def test_guarded_effects_are_local_and_export_as_guarded_openqasm() -> None:
    from flagquantum.runtime.dynamic.conformance import (
        LOCAL_TRAJECTORY_FEATURES,
        QISKIT_AER_DYNAMIC_FEATURES,
        assess_dynamic_features,
    )
    from flagquantum.runtime.dynamic.dialects.openqasm3 import export_dynamic_qasm3

    _, circuit = _binding()
    assert assess_dynamic_features(circuit, LOCAL_TRAJECTORY_FEATURES).compatible
    report = assess_dynamic_features(circuit, QISKIT_AER_DYNAMIC_FEATURES)
    assert "conditional_gate_unsupported:measure" in report.blockers
    assert "conditional_gate_unsupported:reset" in report.blockers
    text = export_dynamic_qasm3(circuit)
    assert "if (c[3] == true) { c[2] = measure q[1]; }" in text
    assert "if (c[3] == true) { reset q[2]; }" in text


def test_inline_hmr_alias_snapshots_before_the_write() -> None:
    source = (
        'INFO_TAG "@flagquantum:reversible-phase-v1 wires=1 bits=1"\n'
        "BIT_STORE0 b0\nBIT_INVERT b0\nHMR q0 b0 if b0\n"
    )
    gates = bind_reversible_phase_execution(decode_kmx(source))
    circuit = DynamicCircuit(1, dtype=torch.complex128).append(FieldOperation(1, gates))
    result = run_dynamic(circuit, shots=32, seed=41, strategy="trajectory")
    assert set(result.classical_bits[:, 0].tolist()) == {0, 1}
    assert torch.all(result.classical_bits[:, 1] == 1)
    assert torch.all(result.final_states.reshape(32, -1)[:, 1] == 0)


def test_guarded_measurement_requires_an_existing_destination() -> None:
    guard = ClassicalXor((), 0, constant=True).instruction()
    metadata = {"is_dynamic": True, "classical_bit": 1, "conditions": ((0, 1),)}
    ordinary = Instruction("measure", (0,), metadata=metadata)
    with pytest.raises(ValidationError, match="not constructible"):
        validate_classical_control((guard, ordinary), n_qubits=1)
    internal = Instruction(
        "measure", (0,), metadata={**metadata, "guarded_effect": True}
    )
    with pytest.raises(ValidationError, match="requires a defined result"):
        validate_classical_control((guard, internal), n_qubits=1)
    assert CircuitIR.from_dict(_binding()[1].to_ir().to_dict()) == _binding()[1].to_ir()


@pytest.mark.ftqc_reference
def test_local_binding_matches_the_native_guarded_branch_table() -> None:
    km = pytest.importorskip("kickmix")
    assert hasattr(km, "Simulator")
    shots = 64
    native = km.Simulator(batch_size=shots, seed=41)
    native.write_across_shots(km.q(1), (1 << shots) - 1)
    native.do(km.Circuit(_SOURCE))
    wires = [native.read_across_shots(km.q(q), out=int) for q in range(4)]
    bits = native.read_across_shots(km.b(0), out=int)
    expected = {
        (
            sum(((column >> shot) & 1) << (3 - q) for q, column in enumerate(wires)),
            (bits >> shot) & 1,
            native.read_shot_phase(shot),
        )
        for shot in range(shots)
    }
    gates = bind_reversible_phase_execution(decode_kmx(_SOURCE))
    circuit = (
        DynamicCircuit(4, dtype=torch.complex128).x(1).append(FieldOperation(4, gates))
    )
    result = run_dynamic(circuit, shots=shots, seed=41, strategy="trajectory")
    states = result.final_states.reshape(shots, -1)
    actual = {
        (int(index), int(bit), int(row[index].real < 0))
        for row, index, bit in zip(
            states, states.abs().argmax(dim=1), result.classical_bits[:, 0], strict=True
        )
    }
    assert actual == expected == {(4, 0, 0), (1, 0, 0), (9, 1, 1)}


def test_declared_classical_inputs_survive_target_binding() -> None:
    source = (
        'INFO_TAG "@flagquantum:reversible-phase-v1 wires=1 bits=2 inputs=0"\n'
        "X q0 if b0\n"
    )
    from flagquantum.compiler.formats.kmx import encode_kmx

    program = decode_kmx(source)
    assert encode_kmx(program) == source
    assert program.input_bits == (0,)
    with pytest.raises(ValueError, match="classical inputs"):
        decode_kmx(source.replace("inputs=0", "inputs=0,0"))
    gates = bind_reversible_phase_execution(program, classical_offset=2)
    initialize = ClassicalXor((), 2, constant=True).instruction()
    circuit = DynamicCircuit(1).append(FieldOperation(1, (initialize, *gates)))
    result = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
    assert torch.all(result.samples[:, 0] == 1)
    assert torch.all(result.classical_bits[:, 2] == 1)
    assert torch.all(result.classical_bits[:, 3] == 0)
