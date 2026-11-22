"""Exercise complete original exact phase conversion and its native consumers."""

import json
from collections import Counter, defaultdict
from fractions import Fraction
from math import pi
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats._physical_operation_statistics import (
    PhysicalOperationStatistics,
)
from flagquantum.compiler.formats.kmb import KmxRecord, encode_kmb
from flagquantum.compiler.formats.kmx_circuit import KmxCircuit, read_circuit_file
from flagquantum.ecosystem._phase_angle import phase_angle_from_half_turns
from flagquantum.simulation.physical_basis_shots import (
    PackedBasisState,
    apply_packed_basis_records,
)
from tests.unit.circuit_asset_fixtures import instructions
from tests.unit.fixtures.generate_phase_angle_objects import cases

pytestmark = pytest.mark.unit
_DATA = json.loads(
    (Path(__file__).with_name("data") / "phase_angle_objects.json").read_text()
)
_VALUES = dict(cases())
_ERRORS = {
    "TypeError": TypeError,
    "ValueError": ValueError,
    "OverflowError": OverflowError,
    "ZeroDivisionError": ZeroDivisionError,
}


@pytest.mark.parametrize("row", _DATA["rows"])
def test_both_original_consumers_word_for_word(row: list[object]) -> None:
    name, units, error, message, source = row
    assert isinstance(name, str) and isinstance(source, str)
    value = _VALUES[name]
    if error is not None:
        assert isinstance(error, str) and isinstance(message, str)
        original = phase_angle_from_half_turns(Fraction(1, 4))
        with pytest.raises(_ERRORS[error]) as captured:
            phase_angle_from_half_turns(value)
        assert str(captured.value) == message
        assert original.half_turns() == Fraction(1, 4)
        assert source == ""
        return
    assert isinstance(units, str)
    angle = phase_angle_from_half_turns(value)
    assert angle.units == int(units)
    # The builder consumer retains exactly one phase instruction, including zero.
    record = KmxRecord("Z_POW", (0,), angle=angle)
    assert record.to_kmx() == source
    circuit = KmxCircuit((), (record,))
    restored = read_circuit_file(encode_kmb(circuit.to_kmx()))
    assert restored == circuit
    # Seed the existing packed phase state with the same converted ring value.
    state = PackedBasisState(defaultdict(int, {0: 1}), defaultdict(int), [angle.units])
    executed: Counter[str] = Counter()
    apply_packed_basis_records(circuit, state, lambda: 0, counts=executed)
    assert executed == {"Z_POW": 1}
    assert state.phases == [(angle.units * 2) % (1 << 128)]
    counts = PhysicalOperationStatistics.from_circuit(circuit)
    assert counts["Z_POW"] == 1
    assert counts.z_pow_clifford_hierarchy_counts == {
        angle.clifford_hierarchy_level(): 1
    }
    inputs = torch.tensor([0.3 + 0.4j, 0.5 + (0.5**0.5) * 1j], dtype=torch.complex128)
    actual = (
        Circuit(1, inputs=inputs, dtype=torch.complex128)
        .append(FieldOperation(1, instructions(circuit)))
        .state()
        .flatten()
    )
    # Native complex execution uses a numerical boundary after exact storage.
    expected = inputs.clone()
    phase = torch.exp(
        torch.tensor(
            1j * pi * float(Fraction(int(units), 1 << 127)), dtype=torch.complex128
        )
    )
    expected[1] *= phase
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert state.qubits[0] == 1


def test_rounding_identity_and_different_input_semantics() -> None:
    assert phase_angle_from_half_turns(0.1) != phase_angle_from_half_turns("0.1")
    assert phase_angle_from_half_turns("0.1") == phase_angle_from_half_turns(
        Fraction(1, 10)
    )
    assert phase_angle_from_half_turns(Fraction(1, 1 << 128)).units == 0
    assert phase_angle_from_half_turns(Fraction(3, 1 << 128)).units == 2
    assert phase_angle_from_half_turns(2 - Fraction(1, 1 << 128)).units == 0
    assert phase_angle_from_half_turns(-Fraction(1, 1 << 128)).units == 0
