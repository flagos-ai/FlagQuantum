"""Compare the locked Z-enable and X-output matrix through register equality."""

import json
from itertools import product
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.comparison.equality_binding import (
    bound_quantum_equal_ir,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient as Bit,
)
from flagquantum.compiler.registers.primitives._mixed_gates import _XBasisTarget
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.fixtures.kickmix_source_oracle import normalized_stream
from tests.unit.test_kickmix_bit_rotation import _certify

pytestmark = pytest.mark.unit
_SOURCE = json.loads(
    (Path(__file__).parent / "fixtures" / "kickmix_control_binding.json").read_text()
)


@pytest.mark.parametrize("key", _SOURCE)
def test_source_control_binding(key: str) -> None:
    output_mode, enable_mode = key.split("/")
    data = 2 + int(output_mode == "quantum") + int(enable_mode == "quantum")
    n_inputs = int(output_mode == "classical") + int(enable_mode == "classical")
    outputs: dict[str, int | _XBasisTarget] = {
        "quantum": 2,
        "phase": _XBasisTarget(True),
        "skip": _XBasisTarget(False),
        "classical": _XBasisTarget(Bit(0)),
    }
    controls: dict[str, int | bool | Bit] = {
        "quantum": data - 1,
        "phase": True,
        "skip": False,
        "classical": Bit(n_inputs - 1 if n_inputs else 0),
    }
    body = bound_quantum_equal_ir(
        (1, 0),
        (True, False),
        outputs[output_mode],
        (data, data + 1),
        control=controls[enable_mode],
        measurement_bit=n_inputs,
    )
    slots = n_inputs + int(bool(n_inputs and body))
    prefix = tuple(ClassicalXor((), bit).instruction() for bit in range(slots))
    program = lower_reversible_phase(CircuitIR(data + 2, (*prefix, *body)))
    actual = normalized_stream("\n".join(encode_kmx(program).splitlines()[1 + slots :]))
    assert actual == [(gate, tuple(guards)) for gate, guards in _SOURCE[key]]

    labels = torch.arange(1 << data)
    basis = labels.to(torch.float64)
    amplitudes = torch.complex(basis + 1, 3 * basis - 5)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (data + 2), dtype=torch.complex128)
    initial[labels << 2] = amplitudes
    for values in product((False, True), repeat=n_inputs):
        matches = labels >> (data - 2) == 1  # q1 is the high bit of the input word.
        if enable_mode == "quantum":
            matches &= (labels & 1).bool()
        elif enable_mode == "skip" or (enable_mode == "classical" and not values[-1]):
            matches[:] = False
        target_labels = labels
        signed = amplitudes
        phase = None
        if output_mode == "quantum":
            target_labels = labels ^ (matches.to(torch.int64) << (data - 3))
        elif output_mode == "phase" or (output_mode == "classical" and values[0]):
            phase = matches
            signed = torch.where(matches, -amplitudes, amplitudes)
        expected = torch.zeros_like(initial)
        expected[target_labels << 2] = signed
        initialization = tuple(
            ClassicalXor((), bit, constant=value).instruction()
            for bit, value in enumerate(
                (*values, *((False,) if slots > n_inputs else ()))
            )
        )
        _certify(
            lower_reversible_phase(CircuitIR(data + 2, (*initialization, *body))),
            data,
            target_labels,
            expected_phase=phase,
        )
        circuit = DynamicCircuit(data + 2, inputs=initial, dtype=torch.complex128)
        circuit.append(FieldOperation(data + 2, (*initialization, *body)))
        result = run_dynamic(circuit, shots=24, seed=47, strategy="batched")
        torch.testing.assert_close(
            result.final_states.reshape(24, -1),
            expected.expand(24, -1),
            atol=1e-12,
            rtol=0,
        )
        for bit, value in enumerate(values):
            assert torch.all(result.classical_bits[:, bit] == value)


def test_shared_outer_slot_and_feedback_are_separate() -> None:
    output = _XBasisTarget(Bit(0))
    with pytest.raises(ValueError, match="separate result slot"):
        bound_quantum_equal_ir(
            (1, 0), (True, False), output, (2, 3), control=Bit(0), measurement_bit=0
        )
    body = bound_quantum_equal_ir(
        (1, 0), (True, False), output, (2, 3), control=Bit(0), measurement_bit=1
    )
    assert all(gate.metadata["conditions"].count((0, 1)) == 1 for gate in body)
