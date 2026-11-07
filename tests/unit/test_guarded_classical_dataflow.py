"""Contract fake before integrating guarded classical parity consumers."""

from itertools import product

import pytest

from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import Instruction

pytestmark = pytest.mark.unit


def _reference_assignment(
    values: list[int],
    sources: tuple[int, ...],
    target: int,
    constant: bool,
    conditions: tuple[tuple[int, int], ...],
) -> list[int]:
    """Independent whole-record fake with old-read and inactive-preserve semantics."""
    old = values.copy()
    result = values.copy()
    if all(old[slot] == expected for slot, expected in conditions):
        result[target] = (sum(old[source] for source in sources) + int(constant)) % 2
    return result


def test_contract_fake_exhausts_alias_and_inactive_truth_tables() -> None:
    for bits, target, constant in product(
        product((0, 1), repeat=3), range(3), (False, True)
    ):
        for sources in ((), (0,), (1, 2), (0, 1, 0), (target, target, 2)):
            conditions = ((0, 1), (1, 0))
            result = _reference_assignment(
                list(bits), sources, target, constant, conditions
            )
            assert result[target] == (
                (sum(bits[source] for source in sources) + constant) % 2
                if bits[:2] == (1, 0)
                else bits[target]
            )
            assert all(
                result[slot] == bits[slot] for slot in range(3) if slot != target
            )


@pytest.mark.ftqc_reference
def test_locked_native_guard_snapshots_match_independent_classical_fake() -> None:
    km = pytest.importorskip("kickmix")
    assert km.__version__ in {"0.1", "0.1.dev0"} and hasattr(km, "Simulator")
    text = "\n".join(
        (
            "PUSH_CONDITION if b0",
            "PUSH_CONDITION if b1",
            "BIT_INVERT b0 if b2",
            "BIT_INVERT b3",
            "POP_CONDITION",
            "BIT_INVERT b3",
            "POP_CONDITION",
        )
    )
    for bits in product((0, 1), repeat=4):
        sim = km.Simulator(batch_size=1, seed=31)
        for slot, value in enumerate(bits):
            sim.write_across_shots(km.b(slot), value)
        sim.do(km.Circuit(text))
        expected = list(bits)
        outer, inner = bits[0], bits[1]
        expected = (
            _reference_assignment(expected, (0, 2), 0, False, ())
            if outer and inner
            else expected
        )
        expected = (
            _reference_assignment(expected, (3,), 3, True, ())
            if outer and inner
            else expected
        )
        expected = (
            _reference_assignment(expected, (3,), 3, True, ()) if outer else expected
        )
        assert [
            sim.read_across_shots(km.b(slot), out=int) for slot in range(4)
        ] == expected


def _guarded(assignment: object, conditions: tuple[tuple[int, int], ...]) -> object:

    instruction = assignment.instruction()
    return Instruction(
        instruction.name,
        instruction.wires,
        instruction.params,
        metadata={**instruction.metadata, "conditions": conditions},
    )


@pytest.mark.parametrize("strategy", ["trajectory", "batched"])
@pytest.mark.parametrize("target", [0, 1, 2])
@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize("sources", [(), (0,), (1, 2), (0, 1, 0), (0, 1, 2, 2)])
def test_both_executors_match_independent_fake_with_aliases(
    strategy: str, target: int, constant: bool, sources: tuple[int, ...]
) -> None:
    import torch

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.core.ir import Instruction
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    circuit = DynamicCircuit(4, dtype=torch.complex128)
    for wire in range(3):
        circuit.gate("h", wire)
        circuit.measure(wire, classical_bit=wire)
    circuit.append(
        FieldOperation(
            4,
            (
                _guarded(ClassicalXor(sources, target, constant), ((0, 1), (1, 0))),
                Instruction("x", (3,), metadata={"conditions": ((target, 1),)}),
            ),
        )
    )
    result = run_dynamic(circuit, shots=64, seed=41, strategy=strategy)
    assert result.statistics["measurement_count"] == 3 * 64
    assert result.statistics["reset_count"] == 0
    indices = result.final_states.abs().argmax(dim=1).tolist()
    for index, record in zip(indices, result.classical_bits.tolist(), strict=True):
        old = [(index >> (3 - q)) & 1 for q in range(3)]
        expected = _reference_assignment(
            old, sources, target, constant, ((0, 1), (1, 0))
        )
        assert record == expected
        assert index & 1 == expected[target]


def test_density_branches_and_guard_sources_are_retained() -> None:
    import torch

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.core.ir import Instruction
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.simulation.density_matrix import density_matrix_from_ir

    circuit = DynamicCircuit(3, dtype=torch.complex128)
    for wire in range(2):
        circuit.gate("h", wire)
        circuit.measure(wire, classical_bit=wire)
    circuit.append(
        FieldOperation(
            3,
            (
                ClassicalXor((), 2).instruction(),
                _guarded(ClassicalXor((0, 1), 2), ((0, 1),)),
                Instruction("x", (2,), metadata={"conditions": ((2, 1),)}),
            ),
        )
    )
    actual = density_matrix_from_ir(circuit).reshape(8, 8)
    diagonal = torch.zeros(8, dtype=torch.complex128)
    diagonal[torch.tensor([0, 2, 5, 6])] = 0.25
    torch.testing.assert_close(actual, torch.diag(diagonal), atol=2e-14, rtol=2e-14)


def test_conditional_def_use_and_parser_rejections() -> None:
    from flagquantum.core.classical_control import validate_classical_control
    from flagquantum.core.ir import Instruction
    from flagquantum.errors import ValidationError

    zero = ClassicalXor((), 0).instruction()
    with pytest.raises(ValidationError, match="defined destination"):
        validate_classical_control(
            (zero, _guarded(ClassicalXor((), 1), ((0, 1),))), n_qubits=1
        )
    with pytest.raises(ValidationError, match="before definition"):
        validate_classical_control(
            (zero, _guarded(ClassicalXor((), 0), ((1, 1),))), n_qubits=1
        )
    with pytest.raises(ValidationError, match="before definition"):
        validate_classical_control(
            (zero, _guarded(ClassicalXor((1,), 0), ((0, 1),))), n_qubits=1
        )
    for conditions in ((), ((0, 1), (0, 1)), ((1, 1), (0, 1))):
        with pytest.raises(ValidationError):
            _guarded(ClassicalXor((), 0), conditions)
    with pytest.raises(ValidationError):
        Instruction(
            "classical_xor",
            (),
            params={"sources": ()},
            metadata={
                "is_dynamic": True,
                "classical_bit": 0,
                "condition_clauses": (((0, 1),),),
            },
        )


@pytest.mark.parametrize("strategy", ["trajectory", "batched"])
def test_scoped_guard_alias_and_nested_snapshots_match_native_contract(
    strategy: str,
) -> None:
    import torch

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
    from flagquantum.compiler.lowering.reversible_phase_execution import (
        bind_reversible_phase_execution,
    )
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    text = (
        "\n".join(
            (
                'INFO_TAG "@flagquantum:reversible-phase-v1 wires=1 bits=4 inputs=0,1,2,3"',
                "PUSH_CONDITION if b0",
                "PUSH_CONDITION if b1",
                "BIT_INVERT b0 if b2",
                "BIT_INVERT b3",
                "POP_CONDITION",
                "BIT_INVERT b3",
                "POP_CONDITION",
            )
        )
        + "\n"
    )
    program = decode_kmx(text)
    assert encode_kmx(program) == text
    for bits in product((0, 1), repeat=4):
        setup = tuple(
            ClassicalXor((), slot, bool(value)).instruction()
            for slot, value in enumerate(bits)
        )
        gates = bind_reversible_phase_execution(program)
        circuit = DynamicCircuit(1, dtype=torch.complex128).append(
            FieldOperation(1, (*setup, *gates))
        )
        result = run_dynamic(circuit, shots=4, seed=17, strategy=strategy)
        expected = list(bits)
        if bits[0] and bits[1]:
            expected[0] ^= bits[2]
            expected[3] ^= 1
        if bits[0]:
            expected[3] ^= 1
        assert torch.all(result.classical_bits[:, :4] == torch.tensor(expected))
        assert (
            result.statistics["measurement_count"]
            == result.statistics["reset_count"]
            == 0
        )


def test_guarded_dataflow_exports_and_typed_route_refuses_merge() -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
    from flagquantum.core.ir.program.lowering import circuit_to_program
    from flagquantum.core.ir.program.model import UNSUPPORTED_WITH_DIAGNOSTICS
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.conformance import (
        LOCAL_TRAJECTORY_FEATURES,
        assess_dynamic_features,
    )
    from flagquantum.runtime.dynamic.dialects.openqasm3 import export_dynamic_qasm3

    circuit = DynamicCircuit(1).append(
        FieldOperation(
            1,
            (
                ClassicalXor((), 0).instruction(),
                _guarded(ClassicalXor((0,), 0, True), ((0, 1),)),
            ),
        )
    )
    assert assess_dynamic_features(circuit, LOCAL_TRAJECTORY_FEATURES).compatible
    assert "if (c[0] == true) { c[0] = c[0] ^ true; }" in export_dynamic_qasm3(circuit)
    conversion = circuit_to_program(circuit.to_ir())
    assert conversion.state == UNSUPPORTED_WITH_DIAGNOSTICS
    assert "conditional" in conversion.diagnostics[0].message
    assert lower_reversible_phase(circuit.to_ir()).n_bits == 1


def test_guarded_parity_example() -> None:
    from examples.algorithms.guarded_classical_parity import main

    main()


@pytest.mark.parametrize("strategy", ["trajectory", "batched"])
def test_classical_guards_consume_no_rng_or_quantum_work(strategy: str) -> None:
    import torch

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    def make(guarded: bool) -> DynamicCircuit:
        circuit = DynamicCircuit(2, dtype=torch.complex128)
        for wire in range(2):
            circuit.gate("h", wire)
            circuit.measure(wire, classical_bit=wire)
        gates = (ClassicalXor((), 2).instruction(),)
        if guarded:
            gates = (*gates, _guarded(ClassicalXor((0, 1), 2, True), ((0, 1),)))
        return circuit.append(FieldOperation(2, gates))

    reference = run_dynamic(make(False), shots=64, seed=41, strategy=strategy)
    actual = run_dynamic(make(True), shots=64, seed=41, strategy=strategy)
    torch.testing.assert_close(
        actual.final_states, reference.final_states, rtol=0, atol=0
    )
    assert torch.equal(actual.classical_bits[:, :2], reference.classical_bits[:, :2])
    assert (
        actual.statistics["measurement_count"]
        == reference.statistics["measurement_count"]
    )


@pytest.mark.parametrize("missing_source", [False, True])
def test_density_refuses_undefined_reads_even_in_inactive_assignments(
    missing_source: bool,
) -> None:
    from flagquantum.core.ir import CircuitIR
    from flagquantum.errors import ValidationError
    from flagquantum.simulation.density_matrix import density_matrix_from_ir

    assignment = ClassicalXor((9,), 0) if missing_source else ClassicalXor((), 1)
    ir = CircuitIR(
        1, (ClassicalXor((), 0).instruction(), _guarded(assignment, ((0, 1),)))
    )
    with pytest.raises(ValidationError, match="before definition|defined destination"):
        density_matrix_from_ir(ir)
