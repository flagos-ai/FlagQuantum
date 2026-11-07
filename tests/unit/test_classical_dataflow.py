"""Independent contract fake and primitive truth-table conformance."""

import itertools
from dataclasses import replace
from pathlib import Path

import pytest

from flagquantum.core.classical_dataflow import ClassicalXor, parse_classical_xor
from flagquantum.core.ir import CircuitIR, Instruction, IRValidationError
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import ValidationError

pytestmark = pytest.mark.unit


def _reference_xor(
    values: list[int], sources: tuple[int, ...], target: int, constant: bool
) -> list[int]:
    """Reference fake: count old set bits and assign modulo two."""
    result = values.copy()
    result[target] = (sum(values[bit] for bit in sources) + int(constant)) % 2
    return result


@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize(
    "sources,target",
    [((), 2), ((0,), 2), ((0, 1), 2), ((0, 0, 1), 2), ((0, 1), 0), ((0, 0, 1), 0)],
)
def test_xor_contract_against_independent_fake(
    sources: tuple[int, ...], target: int, constant: bool
) -> None:
    operation = ClassicalXor(sources, target, constant)
    for bits in itertools.product((0, 1), repeat=3):
        values = list(bits)
        result = operation.evaluate(values)
        values[target] = result
        assert values == _reference_xor(list(bits), sources, target, constant)
    instruction = operation.instruction()
    assert parse_classical_xor(instruction) == operation
    ir = CircuitIR(1, (instruction,))
    assert CircuitIR.from_dict(ir.to_dict()).instructions == ir.instructions


def test_undefined_sources_are_not_cancelled_before_validation() -> None:
    with pytest.raises(ValidationError, match="undefined"):
        ClassicalXor((0, 0), 1).evaluate([-1, 0])
    with pytest.raises(ValidationError, match="nonnegative"):
        ClassicalXor((True,), 0)


def _dataflow_circuit() -> DynamicCircuit:
    import torch

    from flagquantum.dynamic import DynamicCircuit

    circuit = DynamicCircuit(4, dtype=torch.complex128)
    for wire in range(3):
        circuit.h(wire).measure(wire, classical_bit=wire)
    return circuit.compose(
        CircuitIR(
            4,
            (
                ClassicalXor((0, 1), 3).instruction(),
                ClassicalXor((3, 2), 3).instruction(),
                ClassicalXor((), 4).instruction(),
                ClassicalXor((1, 1), 4).instruction(),
                ClassicalXor((0, 3), 0).instruction(),
                ClassicalXor((3,), 3, constant=True).instruction(),
                Instruction("x", (3,), metadata={"conditions": ((3, 0),)}),
                ClassicalXor((3,), 3, constant=True).instruction(),
            ),
        )
    )


@pytest.mark.integration
@pytest.mark.parametrize("strategy", ["trajectory", "batched"])
def test_shared_xor_executes_in_both_trajectory_modes(strategy: str) -> None:
    import torch

    from flagquantum.runtime.dynamic.execution import run_dynamic

    result = run_dynamic(_dataflow_circuit(), shots=128, seed=31, strategy=strategy)
    assert result.statistics["measurement_count"] == 3 * 128
    states = result.final_states.reshape(128, -1)
    basis = states.abs().argmax(dim=1)
    assert set((basis >> 1).tolist()) == set(range(8))
    for index, classical in zip(
        basis.tolist(), result.classical_bits.tolist(), strict=True
    ):
        a, b, c, target = ((index >> shift) & 1 for shift in reversed(range(4)))
        parity = a ^ b ^ c
        assert classical == [a ^ parity, b, c, parity, 0]
        assert target == parity
    torch.testing.assert_close(
        states.abs().sum(dim=1), torch.ones(128, dtype=torch.float64)
    )


@pytest.mark.integration
def test_density_ensemble_keeps_transitive_xor_sources() -> None:
    import torch

    from flagquantum.simulation.density_matrix import density_matrix_from_ir

    actual = density_matrix_from_ir(_dataflow_circuit()).reshape(16, 16)
    diagonal = torch.zeros(16, dtype=torch.complex128)
    for value in range(8):
        diagonal[(value << 1) | (value.bit_count() % 2)] = 1 / 8
    torch.testing.assert_close(actual, torch.diag(diagonal), rtol=0, atol=1e-12)


def test_xor_program_ssa_round_trip_and_type_checks() -> None:
    from flagquantum.core.ir.program.lowering import (
        circuit_to_program,
        program_to_circuit,
    )
    from flagquantum.core.ir.program.model import SUPPORTED_EXACT
    from flagquantum.core.ir.program.verifier import verify_program_record

    ir = _dataflow_circuit().to_ir()
    conversion = circuit_to_program(ir)
    assert conversion.state == SUPPORTED_EXACT
    record = conversion.record
    assert record is not None
    assert not verify_program_record(record)
    operations = record.functions[0].blocks[0].operations
    xor_results = [op.results[0] for op in operations if op.opcode == "bool_xor"]
    assert operations[-3].condition_clauses[0][0].value == xor_results[-2]
    assert sum(op.opcode == "bool_one" for op in operations) == 2
    restored = program_to_circuit(record)
    assert restored.state == SUPPORTED_EXACT
    assert restored.canonical_payload == ir.to_dict()
    entry = record.functions[0].blocks[0]
    position = next(
        i for i, op in enumerate(entry.operations) if op.opcode == "bool_xor"
    )
    changed = replace(
        entry.operations[position], operands=(entry.arguments[0], entry.arguments[1])
    )
    bad_entry = replace(
        entry,
        operations=(
            *entry.operations[:position],
            changed,
            *entry.operations[position + 1 :],
        ),
    )
    bad_record = replace(
        record, functions=(replace(record.functions[0], blocks=(bad_entry,)),)
    )
    assert any(
        d.code == "program.operand_kind" for d in verify_program_record(bad_record)
    )


def test_validation_rejects_undefined_sources_and_invalid_instruction_shapes() -> None:
    from flagquantum.core.classical_control import validate_classical_control

    for sources in ((0,), (0, 0)):
        with pytest.raises(ValidationError, match="before definition"):
            validate_classical_control(
                (ClassicalXor(sources, 0).instruction(),), n_qubits=1
            )
    assert (
        validate_classical_control((ClassicalXor((), 7).instruction(),), n_qubits=1)
        == 8
    )
    with pytest.raises(ValidationError):
        Instruction(
            "classical_xor",
            (0,),
            params={"sources": ()},
            metadata={"is_dynamic": True, "classical_bit": 0},
        )
    with pytest.raises(ValidationError):
        Instruction(
            "classical_xor",
            (),
            params={"sources": ()},
            metadata={"is_dynamic": True, "classical_bit": 0, "conditions": ((1, 1),)},
        )
    with pytest.raises(IRValidationError, match="at least one"):
        Instruction("x", ())


def test_qasm3_emits_boolean_assignment() -> None:
    from flagquantum.runtime.dynamic.dialects.openqasm3 import export_dynamic_qasm3

    text = export_dynamic_qasm3(_dataflow_circuit())
    assert "c[3] = c[0] ^ c[1];" in text
    assert "c[3] = c[3] ^ c[2];" in text
    assert "c[4] = false;" in text
    assert text.count("c[3] = c[3] ^ true;") == 2


@pytest.mark.ftqc_reference
def test_kmx_classical_assignment_round_trip_and_native_execution(
    tmp_path: Path,
) -> None:
    km = pytest.importorskip("kickmix")
    from flagquantum.compiler.artifacts.reversible_phase import (
        ReversiblePhaseOperation,
        ReversiblePhaseProgram,
    )
    from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
    from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase

    assignments = (
        ClassicalXor((0, 1), 2),
        ClassicalXor((2, 0), 2),
        ClassicalXor((0, 0, 1), 0),
        ClassicalXor((0, 1), 0),
        ClassicalXor((0,), 0, constant=True),
    )
    hmr = []
    for i in range(2):
        hmr.extend(
            (
                Instruction("h", (i,)),
                Instruction(
                    "measure", (i,), metadata={"is_dynamic": True, "classical_bit": i}
                ),
                Instruction("x", (i,), metadata={"conditions": ((i, 1),)}),
            )
        )
    lowered = lower_reversible_phase(
        CircuitIR(2, (*hmr, *(op.instruction() for op in assignments)))
    )
    assert lowered.operations[-1] == ReversiblePhaseOperation(
        "bit_invert", (), result_bit=0
    )
    # Replace the measurement source with an independent deterministic classical
    # producer. The compiled XOR consumer stream is unchanged for all truth rows.
    for bits in itertools.product((0, 1), repeat=2):
        prepared = [
            ReversiblePhaseOperation("bit_store0", (), result_bit=i) for i in range(2)
        ]
        prepared.extend(
            ReversiblePhaseOperation("bit_invert", (), result_bit=i)
            for i, bit in enumerate(bits)
            if bit
        )
        program = ReversiblePhaseProgram(2, 3, (*prepared, *lowered.operations[2:]))
        path = tmp_path / "xor.kmx"
        path.write_text(encode_kmx(program))
        assert decode_kmx(path.read_text()) == program
        sim = km.Simulator(batch_size=1, seed=0)
        sim.do(km.Circuit.from_file(str(path)))
        values = [*bits, 0]
        for operation in assignments:
            values[operation.target] = operation.evaluate(values)
        assert [sim.read_across_shots(km.b(i), out=int) for i in range(3)] == values


@pytest.mark.parametrize("strategy", ["trajectory", "batched"])
def test_classical_work_preserves_rng_stream_and_measurement_counters(
    strategy: str,
) -> None:
    import torch

    from flagquantum.runtime.dynamic.execution import run_dynamic

    baseline = DynamicCircuit(2).h(0).measure(0).h(1).measure(1)
    candidate = baseline.copy().compose(
        CircuitIR(2, (ClassicalXor((0, 1), 2).instruction(),))
    )
    expected = run_dynamic(baseline, shots=64, seed=19, strategy=strategy)
    actual = run_dynamic(candidate, shots=64, seed=19, strategy=strategy)
    torch.testing.assert_close(
        actual.final_states, expected.final_states, rtol=0, atol=0
    )
    torch.testing.assert_close(actual.samples, expected.samples, rtol=0, atol=0)
    torch.testing.assert_close(actual.classical_bits[:, :2], expected.classical_bits)
    assert (
        actual.statistics["measurement_count"]
        == expected.statistics["measurement_count"]
    )


def test_restricted_provider_refuses_unimplemented_classical_computation() -> None:
    from types import SimpleNamespace

    from flagquantum.runtime.dynamic.conformance import (
        LOCAL_TRAJECTORY_FEATURES,
        QISKIT_AER_DYNAMIC_FEATURES,
        assess_dynamic_features,
    )
    from flagquantum.runtime.dynamic.deployment import (
        _assess_dynamic_package_compatibility,
    )
    from flagquantum.runtime.dynamic.dialects.braket_iqm import (
        export_braket_iqm_dynamic_qasm3,
    )

    circuit = (
        DynamicCircuit(2)
        .measure(0)
        .compose(CircuitIR(2, (ClassicalXor((0,), 1).instruction(),)))
    )
    assert assess_dynamic_features(circuit, LOCAL_TRAJECTORY_FEATURES).compatible
    report = assess_dynamic_features(circuit, QISKIT_AER_DYNAMIC_FEATURES)
    assert (
        not report.compatible and "classical_computation_unsupported" in report.blockers
    )
    with pytest.raises(ValueError, match="classical_computation_is_unsupported"):
        export_braket_iqm_dynamic_qasm3(circuit, qubit_groups=((0, 1),))
    backend = SimpleNamespace(
        n_qubits=2,
        supports_openqasm=True,
        supports_dynamic_circuits=True,
        dynamic_dialect="openqasm3",
    )
    package = _assess_dynamic_package_compatibility(circuit, backend)
    assert not package.compatible
    assert "backend_classical_computation_is_unqualified" in package.blockers
