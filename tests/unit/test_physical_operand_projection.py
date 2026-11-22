"""Compare complete scalar source identities and normal typed phase execution."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.artifacts._operand_tags import (
    MAX_NUM_BITS,
    MAX_NUM_QUBITS,
    MAX_NUM_REGISTERS,
    PAYLOAD_MASK,
    TAG_MASK,
    format_operand_tag,
    is_scalar_tag,
)
from flagquantum.compiler.artifacts.operand_identity import OperandIdentity, OperandKind
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.comparison.equality_binding import (
    bound_quantum_equal_ir,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.physical_basis_shots import (
    PackedBasisState,
    apply_packed_basis_records,
)
from flagquantum.simulation.physical_operation_counters import PhysicalOperationCounters

pytestmark = pytest.mark.unit
_DATA = json.loads(
    (Path(__file__).with_name("data") / "scalar_identities.json").read_text()
)
_KINDS = {
    "b": OperandKind.CLASSICAL,
    "q": OperandKind.QUANTUM,
    "xb": OperandKind.X_CLASSICAL,
    "r": OperandKind.REGISTER,
}
_TAG_NAMES = {
    "BOOL_VAL": "BOOLEAN",
    "BIT_ID": "CLASSICAL",
    "QUBIT_ID": "QUANTUM",
    "X_BOOL_VAL": "X_BOOLEAN",
    "X_BIT_ID": "X_CLASSICAL",
    "MAY_BE_MIXED_BUT_IS_ALL_XLIKE": "MIXED_XLIKE",
    "MAY_BE_MIXED_BUT_IS_ALL_ZLIKE": "MIXED_ZLIKE",
    "MAY_BE_MIXED": "MIXED",
}


@pytest.mark.parametrize("row", _DATA["source_rows"])
def test_unchanged_scalar_source_observation(row: str) -> None:
    fields = row.split("\t")
    family = fields[0]
    if family == "limits":
        assert tuple(map(int, fields[1:])) == (
            TAG_MASK,
            PAYLOAD_MASK,
            MAX_NUM_QUBITS,
            MAX_NUM_BITS,
            MAX_NUM_REGISTERS,
        )
    elif family == "error":
        _, name, value, _message = fields
        with pytest.raises(ValueError, match="family range"):
            OperandIdentity(_KINDS[name], int(value))
    elif family in ("b", "q", "xb"):
        _, index, word, payload, text, streamed, default, equal, less, more = fields
        item = OperandIdentity(_KINDS[family], int(index))
        zero = OperandIdentity(item.kind, 0)
        assert item.physical_word == int(word)
        assert item.value == int(payload)
        assert item.diagnostic_text == text == streamed
        assert zero.physical_word == int(default)
        assert (item == OperandIdentity(item.kind, item.value)) == bool(int(equal))
        assert (item < zero) == bool(int(less))
        assert (zero < item) == bool(int(more))
    elif family == "r":
        _, index, word, text, streamed, equal, more = fields
        item = OperandIdentity(OperandKind.REGISTER, int(index))
        assert item.physical_word == int(word) == item.value
        assert item.diagnostic_text == text == streamed
        assert (item == OperandIdentity(item.kind, item.value)) == bool(int(equal))
        assert (OperandIdentity(item.kind, 0) < item) == bool(int(more))
    elif family == "h":
        _, index, xword, zword = fields
        z = OperandIdentity(OperandKind.CLASSICAL, int(index))
        x = z.conjugated_by_h()
        assert x.kind is OperandKind.X_CLASSICAL
        assert (x.physical_word, x.conjugated_by_h().physical_word) == (
            int(xword),
            int(zword),
        )
        assert x.conjugated_by_h() == z
    elif family == "kets":
        plus_ket = OperandIdentity(OperandKind.X_BOOLEAN, 0)
        minus_ket = OperandIdentity(OperandKind.X_BOOLEAN, 1)
        assert tuple(map(int, fields[1:])) == (
            plus_ket.physical_word,
            minus_ket.physical_word,
            plus_ket.physical_word,
        )
    elif family == "xbool":
        _, sign, word, text, streamed, minus, plus, conjugated, equal = fields
        item = OperandIdentity(OperandKind.X_BOOLEAN, int(sign))
        assert item.physical_word == int(word)
        assert item.diagnostic_text == text == streamed
        assert bool(item) == bool(int(minus))
        assert (not bool(item)) == bool(int(plus))
        z = item.conjugated_by_h()
        assert z.kind is OperandKind.BOOLEAN and bool(z) == bool(int(conjugated))
        assert z.conjugated_by_h() == item
        assert (item == OperandIdentity(item.kind, item.value)) == bool(int(equal))
    else:
        assert family == "tag"
        _, tag, scalar, source_text = fields
        assert is_scalar_tag(int(tag)) == bool(int(scalar))
        if "::" in source_text:
            expected = "OperandTag." + _TAG_NAMES[source_text.split("::")[1]]
        else:
            expected = source_text.replace("QXZTypeTag8", "OperandTag")
        assert format_operand_tag(int(tag)) == expected


@pytest.mark.parametrize("tag", [-1, 256, True, 1.0, "1", None])
def test_tag_admission_requires_an_original_unsigned_byte(tag: int) -> None:
    for function in (is_scalar_tag, format_operand_tag):
        with pytest.raises(ValueError, match="unsigned byte"):
            function(tag)


@pytest.mark.parametrize("kind", [OperandKind.QUANTUM, OperandKind.REGISTER])
def test_h_basis_conversion_rejects_physical_quantum_and_register_ids(
    kind: OperandKind,
) -> None:
    with pytest.raises(ValueError, match="classical value"):
        OperandIdentity(kind, 0).conjugated_by_h()


def test_boolean_words_and_existing_identity_contract_are_preserved() -> None:
    for sign in (0, 1):
        boolean = OperandIdentity(OperandKind.BOOLEAN, sign)
        assert boolean.physical_word == sign
        assert boolean.diagnostic_text == str(bool(sign)).lower()
        assert boolean.conjugated_by_h().physical_word == (3 << 29) | sign
        assert repr(boolean) == f"OperandIdentity(OperandKind.BOOLEAN, {sign})"
    assert OperandIdentity(OperandKind.REGISTER, 1 << 31).physical_word == 1 << 31
    assert is_scalar_tag(5) is False  # Native REGISTER kind is outside the QC tags.


@pytest.mark.parametrize("sign", [False, True])
def test_conjugated_identity_reaches_append_with_exact_phase_and_counts(
    sign: bool,
) -> None:
    q = OperandIdentity(OperandKind.QUANTUM, 0)
    classical = OperandIdentity(OperandKind.CLASSICAL, 0)
    target = classical.conjugated_by_h()
    gates = bound_quantum_equal_ir(q, 1, target)
    amplitudes = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex128)
    amplitudes /= amplitudes.norm()
    prefix = ClassicalXor((), classical.value, constant=sign).instruction()
    circuit = DynamicCircuit(1, inputs=amplitudes, dtype=torch.complex128)
    circuit.append(FieldOperation(1, (prefix, *gates)))
    program = lower_reversible_phase(circuit.to_ir())
    encoded = encode_kmx(program)
    assert decode_kmx(encoded) == program
    assert Counter(encoded.splitlines()[2 + int(sign) :]) == {
        "X q0 if b0": 4,
        "Z q0 if b0": 1,
    }
    expected = amplitudes.clone()
    if sign:
        expected[1] *= -1
    result = run_dynamic(circuit, shots=2, seed=13, strategy="batched")
    torch.testing.assert_close(
        result.final_states.reshape(2, -1), expected.expand(2, -1), rtol=0, atol=1e-12
    )
    assert torch.all(result.classical_bits[:, 0] == sign)
    assert result.final_states.numel() == 4  # One logical qubit and zero work qubits.
    physical = read_kmx_circuit(encoded)
    # Lane zero starts in |0>, lane one in |1>, with the same classical sign.
    state = PackedBasisState(defaultdict(int, {0: 2}), defaultdict(int), [0, 0])
    tally = PhysicalOperationCounters(2)
    apply_packed_basis_records(physical, state, lambda: 0, observer=tally.observe)
    assert state.qubits[0] == 2 and state.bits[0] == (3 if sign else 0)
    assert state.phases == [0, 0]
    assert state.sign == (2 if sign else 0)
    for lane in range(2):
        counts, levels = tally.snapshot(lane)
        assert dict(counts) == {
            "INFO_TAG": 1,
            "BIT_STORE0": 1,
            "X_IF": 4 if sign else 0,
            "Z_IF": 1 if sign else 0,
            **({"BIT_INVERT": 1} if sign else {}),
        }
        assert levels == ()


@pytest.mark.parametrize(
    "kind,value",
    [("QUANTUM", 0), (2, 0), (OperandKind.QUANTUM, True), (OperandKind.QUANTUM, 0.0)],
)
def test_identity_rejects_untyped_kind_and_payload(
    kind: OperandKind, value: int
) -> None:
    with pytest.raises(ValueError, match="kind and integer payload"):
        OperandIdentity(kind, value)


def test_existing_nonidentity_comparison_and_index_truth() -> None:
    qubit = OperandIdentity(OperandKind.QUANTUM, 0)
    assert qubit.__lt__(object()) is NotImplemented
    assert bool(qubit) is True
