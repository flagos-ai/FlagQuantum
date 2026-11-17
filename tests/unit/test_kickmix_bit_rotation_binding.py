"""Verify the locked rotation binding matrix and native append ownership."""

import json
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.register_bit_rotation import RegisterBitRotation
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.bits.bit_rotation_binding import (
    bound_bit_rotation_ir,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient as Bit,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.fixtures.kickmix_source_oracle import normalized_stream

pytestmark = pytest.mark.unit

_SOURCE = json.loads(
    (Path(__file__).parent / "fixtures" / "kickmix_rotation_binding.json").read_text()
)


@pytest.mark.parametrize("case", _SOURCE)
def test_original_binding_stream(case: str) -> None:
    direction, shift_mode, control_mode = case.split("/")
    controls: dict[str, int | bool | Bit] = {
        "false": False,
        "true": True,
        "classical": Bit(0),
        "quantum": 4,
    }
    control = controls[control_mode]
    data = 4 + int(type(control) is int)
    input_bits = int(isinstance(control, Bit))
    shift: int | tuple[int | bool | Bit, ...]
    if shift_mode == "integer":
        shift = -1
    elif shift_mode == "literal_word":
        shift = (True, False, True)
    elif shift_mode == "bit_word":
        shift = (Bit(input_bits + 1), Bit(input_bits))
        input_bits += 2
    else:
        assert shift_mode == "quantum_word"
        shift = (data + 1, data)
        data += 2
    merged = shift_mode == "quantum_word" and type(control) is int
    clean = (data,) if merged else ()
    prefix = tuple(ClassicalXor((), slot).instruction() for slot in range(input_bits))
    circuit = DynamicCircuit(data + len(clean))
    circuit.append(FieldOperation(circuit.n_qubits, prefix))
    circuit.append(
        RegisterBitRotation(
            (3, 2, 1, 0),
            shift,
            control,
            right=direction == "right",
            clean=clean,
            measurement_bit=input_bits if merged else None,
        )
    )
    program = lower_reversible_phase(circuit.to_ir())
    kmx = encode_kmx(program).splitlines()[1 + len(prefix) :]
    actual = normalized_stream("\n".join(kmx))
    expected = [(opcode, tuple(guards)) for opcode, guards in _SOURCE[case]]
    assert actual == expected


@pytest.mark.parametrize("enable", [False, True])
@pytest.mark.parametrize("low", [False, True])
def test_aliased_control_snapshots_entry_condition(enable: bool, low: bool) -> None:
    target = (0, 1, 2, 3)
    shifted_by = 2 * enable + low if enable else 0
    labels = torch.arange(16)
    rotated = ((labels << shifted_by) | (labels >> (4 - shifted_by))) & 15
    labels_f = labels.to(torch.float64)
    amplitudes = torch.complex(labels_f + 1, 2 * labels_f - 3)
    amplitudes /= amplitudes.norm()
    expected = torch.empty_like(amplitudes)
    expected[rotated] = amplitudes
    prefix = (
        ClassicalXor((), 0, constant=enable).instruction(),
        ClassicalXor((), 1, constant=low).instruction(),
    )
    circuit = DynamicCircuit(4, inputs=amplitudes, dtype=torch.complex128)
    circuit.append(FieldOperation(4, prefix))
    circuit.append(
        RegisterBitRotation(target, (Bit(0), Bit(1)), Bit(0), predicate_bit=2)
    )
    result = run_dynamic(circuit, shots=3, seed=19, strategy="batched")
    torch.testing.assert_close(
        result.final_states.reshape(3, -1), expected.expand(3, -1), atol=1e-12, rtol=0
    )
    assert torch.all(result.classical_bits[:, 0] == enable)
    assert torch.all(result.classical_bits[:, 1] == low)
    assert not torch.any(result.classical_bits[:, 2])


def test_binding_disabled_paths_and_work_admission() -> None:
    target = (0, 1, 2, 3)
    assert bound_bit_rotation_ir(target, (False,) * 61, control=False) == ()
    assert bound_bit_rotation_ir(target, True) == bound_bit_rotation_ir(target, 1)
    with pytest.raises(ValueError, match="distinct nonnegative"):
        bound_bit_rotation_ir((0, 0), control=False)
    assert bound_bit_rotation_ir(target, (Bit(0),), control=Bit(0))
    with pytest.raises(ValueError, match="fresh predicate"):
        bound_bit_rotation_ir(target, (Bit(0), Bit(1)), control=Bit(0))
    with pytest.raises(ValueError, match="fresh result slot"):
        bound_bit_rotation_ir(target, (4, 5), control=6)
