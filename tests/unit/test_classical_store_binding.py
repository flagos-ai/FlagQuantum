"""Compare complete classical store records and native sequential feedback."""

import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient as Bit,
)
from flagquantum.compiler.registers.primitives.classical_store import (
    physical_classical_store,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.fixtures.kickmix_binary_gate_matrix import Port, _port

pytestmark = pytest.mark.unit
_FIXTURE = Path(__file__).with_name("fixtures").joinpath("classical_store_binding.json")


def test_complete_broadcast_store_records_and_refusals() -> None:
    fixture = json.loads(_FIXTURE.read_text())
    cases: list[tuple[Port, object, Port, tuple[int, str] | None]] = fixture["cases"]
    assert len(cases) > 500
    for target, value, control, expected in cases:
        if expected is None:
            with pytest.raises(ValueError):
                physical_classical_store(_port(target), value, control=_port(control))
            continue
        program = physical_classical_store(_port(target), value, control=_port(control))
        text = encode_kmx(program)
        records = "\n".join(text.splitlines()[1:])
        assert (
            len(records.splitlines()),
            hashlib.sha256(records.encode()).hexdigest(),
        ) == tuple(expected), (target, value, control)
        assert decode_kmx(text) == program
        assert all(not operation.qubits for operation in program.operations)


@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("layout", ["self", "cascade", "reduction", "mixed"])
def test_all_classical_inputs_sequential_aliases_and_complex_spectators(
    value: bool, layout: str
) -> None:
    targets = {
        "self": (0, 1),
        "cascade": (0, 1, 0),
        "reduction": (0, 0, 0),
        "mixed": (0, 1, 2),
    }[layout]
    controls: tuple[bool | Bit, ...] = {
        "self": (Bit(0), Bit(0)),
        "cascade": (Bit(0), Bit(0), Bit(1)),
        "reduction": (Bit(1), Bit(2), Bit(3)),
        "mixed": (True, False, Bit(3)),
    }[layout]
    program = physical_classical_store(
        tuple(Bit(slot) for slot in targets), value, control=controls
    )
    amplitude = torch.tensor([1 + 2j, 3 - 4j, -2 + 5j, 7 + 1j], dtype=torch.complex128)
    amplitude /= amplitude.norm()
    for before in product((False, True), repeat=4):
        expected = list(before)
        for target, control in zip(targets, controls, strict=True):
            active = expected[control.slot] if isinstance(control, Bit) else control
            if active:
                expected[target] = value
        setup = tuple(
            ClassicalXor((), slot, bit).instruction() for slot, bit in enumerate(before)
        )
        circuit = DynamicCircuit(2, inputs=amplitude, dtype=torch.complex128)
        circuit.append(
            FieldOperation(2, (*setup, *bind_reversible_phase_execution(program)))
        )
        result = run_dynamic(circuit, shots=1, seed=41, strategy="batched")
        torch.testing.assert_close(
            result.final_states.reshape(-1), amplitude, rtol=0, atol=0
        )
        assert result.classical_bits[0, :4].tolist() == expected
        assert not any(op.opcode in {"r", "hmr"} for op in program.operations)


def test_scalar_target_reduction_and_conditional_old_value() -> None:
    program = physical_classical_store(Bit(0), True, control=(Bit(1), Bit(2), Bit(3)))
    assert [op.result_bit for op in program.operations] == [0, 0, 0]
    assert [op.condition_bit for op in program.operations] == [1, 2, 3]
    assert Counter(op.opcode for op in program.operations) == {"bit_store1": 3}
    for word in range(8):
        setup = tuple(
            ClassicalXor(
                (), slot, bool(word & (1 << (slot - 1))) if slot else False
            ).instruction()
            for slot in range(4)
        )
        circuit = DynamicCircuit(1)
        circuit.append(
            FieldOperation(1, (*setup, *bind_reversible_phase_execution(program)))
        )
        result = run_dynamic(circuit, shots=1, seed=41, strategy="batched")
        assert bool(result.classical_bits[0, 0]) == bool(word)


def test_empty_inputs_sparse_storage_and_value_conversion() -> None:
    empty = physical_classical_store((), True, control=0)
    assert empty.operations == () and empty.n_bits == 0
    skipped = physical_classical_store(Bit(9), False, control=False)
    assert (
        skipped.operations == () and skipped.input_bits == (9,) and skipped.n_bits == 10
    )
    for invalid in ([], [1], "", "false"):
        with pytest.raises(ValueError, match="Boolean-convertible"):
            physical_classical_store(Bit(0), invalid)
    with pytest.raises(ValueError, match="must be classical"):
        physical_classical_store((Bit(0), Bit(1)), True, control=(Bit(2), 0))


def test_boolean_protocol_and_value_validation_precede_port_conversion() -> None:
    class FalseScalar:
        def __bool__(self) -> bool:
            return False

    class LengthOnly:
        def __len__(self) -> int:
            return 0

    class IndexOnly:
        def __index__(self) -> int:
            return 0

    class RaisingScalar:
        def __bool__(self) -> bool:
            raise ValueError("deliberate input failure")

    assert (
        physical_classical_store(Bit(0), FalseScalar()).operations[0].opcode
        == "bit_store0"
    )
    for invalid in (LengthOnly(), IndexOnly()):
        with pytest.raises(ValueError, match="Boolean-convertible"):
            physical_classical_store(Bit(0), invalid)
    with pytest.raises(ValueError, match="Boolean-convertible") as failure:
        physical_classical_store(0, RaisingScalar(), control=False)
    assert isinstance(failure.value.__cause__, ValueError)
    assert str(failure.value.__cause__) == "deliberate input failure"
