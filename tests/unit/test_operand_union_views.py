"""Compare complete original operand unions through native identity contexts."""

import json
from cmath import exp
from collections import Counter, defaultdict
from fractions import Fraction
from math import pi
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.artifacts._operand_tags import PAYLOAD_MASK
from flagquantum.compiler.artifacts._operand_views import (
    OperandDomain,
    admit_operand_identity,
    cast_operand_word,
    default_operand_identity,
    format_operand_word,
)
from flagquantum.compiler.artifacts.operand_identity import OperandIdentity, OperandKind
from flagquantum.compiler.formats._physical_operation_statistics import (
    PhysicalOperationStatistics,
)
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.formats.kmx_circuit import KmxCircuit, read_kmx_circuit
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.primitives.broadcast_binding import broadcast_cx_ir
from flagquantum.compiler.registers.primitives.operand_binding import (
    bind_operand,
    identify_operand,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.physical_basis_shots import (
    PackedBasisState,
    apply_packed_basis_records,
)
from flagquantum.simulation.physical_operation_counters import PhysicalOperationCounters
from tests.unit.fixtures.generate_operand_unions import GateConsumer
from tests.unit.fixtures.kickmix_source_oracle import normalized_stream

pytestmark = pytest.mark.unit
_DATA = json.loads(
    (Path(__file__).with_name("data") / "operand_unions.json").read_text()
)
_ROWS: tuple[str, ...] = tuple(_DATA["source_rows"])


def fields(label: str) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(row.split("\t")[1:]) for row in _ROWS if row.startswith(label + "\t")
    )


def native_diagnostic(text: str) -> str:
    return (
        text.replace("QubitOrXBitOrXBool", "XOperand")
        .replace("QubitOrXZBitOrXZBool", "MixedOperand")
        .replace(".tagged_id=", ".physical_word=")
        .replace(
            "{invalid QubitOrMinusState instance}",
            "{invalid quantum/minus operand instance}",
        )
    )


def identity(word: int) -> OperandIdentity:
    # Only canonical scalar words enter this test helper.
    return OperandIdentity(OperandKind(word >> 29), word & PAYLOAD_MASK)


@pytest.mark.parametrize("row", fields("constructible"))
def test_source_constructor_domains_and_defaults(row: tuple[str, ...]) -> None:
    domain = OperandDomain(row[0])
    for kind, possible in zip(list(OperandKind)[:5], row[1:6], strict=True):
        value = (
            1
            if domain in (OperandDomain.QUANTUM_MINUS, OperandDomain.QUANTUM_TRUE)
            else 0
        )
        item = OperandIdentity(kind, value)
        if int(possible):
            assert admit_operand_identity(item, domain) is item
        else:
            with pytest.raises(ValueError, match="family"):
                admit_operand_identity(item, domain)
    if int(row[7]):
        default = default_operand_identity(domain)
        expected = 3 << 29 if domain is OperandDomain.X_OPERAND else 0
        assert default.physical_word == expected
    else:
        with pytest.raises(ValueError, match="no default"):
            default_operand_identity(domain)
    assert bool(int(row[6])) == (domain is OperandDomain.CLASSICAL_FALSE)


@pytest.mark.parametrize("row", fields("bitbool"))
def test_classical_boolean_union(row: tuple[str, ...]) -> None:
    (
        word,
        payload,
        text,
        streamed,
        is_bool,
        is_bit,
        truth,
        cast,
        eq_bit,
        eq_self,
        eq_bool,
    ) = row
    item = identity(int(word))
    domain = OperandDomain.CLASSICAL_BOOLEAN
    assert admit_operand_identity(item, domain) is item
    assert item.value == int(payload)
    assert format_operand_word(item.physical_word, domain) == text == streamed
    assert (item.kind is OperandKind.BOOLEAN) == bool(int(is_bool))
    assert (item.kind is OperandKind.CLASSICAL) == bool(int(is_bit))
    assert bool(
        cast_operand_word(item.physical_word, domain, OperandKind.BOOLEAN)
    ) == bool(int(truth))
    assert cast_operand_word(
        item.physical_word, domain, OperandKind.CLASSICAL
    ).physical_word == int(cast)
    bit = OperandIdentity(OperandKind.CLASSICAL, int(payload))
    assert (item == bit) == bool(int(eq_bit))
    assert (item == identity(int(word))) == bool(int(eq_self))
    boolean = OperandIdentity(
        OperandKind.BOOLEAN, int(word) if item.kind is OperandKind.BOOLEAN else 0
    )
    assert (item == boolean) == bool(int(eq_bool))


@pytest.mark.parametrize("row", fields("restricted"))
def test_classical_and_quantum_false_unions(row: tuple[str, ...]) -> None:
    (
        index,
        bword,
        is_bit,
        bcast,
        bpayload,
        btext,
        beq,
        qword,
        is_qubit,
        qcast,
        qpayload,
        qtext,
        qeq,
    ) = row
    for domain, word, kind, predicate, cast, payload, text, eq in (
        (
            OperandDomain.CLASSICAL_FALSE,
            bword,
            OperandKind.CLASSICAL,
            is_bit,
            bcast,
            bpayload,
            btext,
            beq,
        ),
        (
            OperandDomain.QUANTUM_FALSE,
            qword,
            OperandKind.QUANTUM,
            is_qubit,
            qcast,
            qpayload,
            qtext,
            qeq,
        ),
    ):
        item = OperandIdentity(kind, int(index))
        assert admit_operand_identity(item, domain) is item
        assert item.physical_word == int(word)
        assert item.value == int(payload) and int(predicate) == 1
        assert cast_operand_word(item.physical_word, domain, kind).physical_word == int(
            cast
        )
        assert format_operand_word(item.physical_word, domain) == text
        assert (item == identity(int(word))) == bool(int(eq))


@pytest.mark.parametrize("row", fields("qubitbit"))
def test_quantum_classical_union_and_false_default(row: tuple[str, ...]) -> None:
    word, payload, is_bit, is_qubit, text, bcast, qcast, equal = row
    domain = OperandDomain.QUANTUM_CLASSICAL
    item = identity(int(word))
    if int(word):
        assert admit_operand_identity(item, domain) is item
    else:
        assert default_operand_identity(domain) == item
        with pytest.raises(ValueError, match="family"):
            admit_operand_identity(item, domain)
    assert item.value == int(payload)
    assert (item.kind is OperandKind.CLASSICAL) == bool(int(is_bit))
    assert (item.kind is OperandKind.QUANTUM) == bool(int(is_qubit))
    assert format_operand_word(int(word), domain) == text
    assert cast_operand_word(
        int(word), domain, OperandKind.CLASSICAL
    ).physical_word == int(bcast)
    assert cast_operand_word(
        int(word), domain, OperandKind.QUANTUM
    ).physical_word == int(qcast)
    assert (item == identity(int(word))) == bool(int(equal))


@pytest.mark.parametrize("row", fields("qubitconstant"))
def test_quantum_true_and_minus_unions(row: tuple[str, ...]) -> None:
    (
        index,
        tword,
        is_true,
        tq,
        tpayload,
        tcast,
        mword,
        is_minus,
        mq,
        mpayload,
        mcast,
        text,
        equal,
    ) = row
    item = OperandIdentity(OperandKind.QUANTUM, int(index))
    for domain, word, is_constant, quantum, payload, cast in (
        (OperandDomain.QUANTUM_TRUE, tword, is_true, tq, tpayload, tcast),
        (OperandDomain.QUANTUM_MINUS, mword, is_minus, mq, mpayload, mcast),
    ):
        assert admit_operand_identity(item, domain) is item
        assert item.physical_word == int(word) and item.value == int(payload)
        assert int(is_constant) == 0 and int(quantum) == 1
        assert cast_operand_word(
            int(word), domain, OperandKind.QUANTUM
        ).physical_word == int(cast)
    assert format_operand_word(item.physical_word, OperandDomain.QUANTUM_MINUS) == text
    assert (item == identity(int(mword))) == bool(int(equal))


@pytest.mark.parametrize("row", fields("z"))
def test_complete_z_union_raw_projections(row: tuple[str, ...]) -> None:
    (
        word,
        payload,
        tag,
        text,
        streamed,
        boolean,
        false,
        true,
        bit,
        quantum,
        truth,
        bcast,
        qcast,
        equal,
    ) = row
    raw = int(word)
    domain = OperandDomain.Z_OPERAND
    assert (raw >> 29, raw & PAYLOAD_MASK) == (int(tag), int(payload))
    assert (
        format_operand_word(raw, domain)
        == native_diagnostic(text)
        == native_diagnostic(streamed)
    )
    assert (int(tag) == 0, raw == 0, raw == 1, int(tag) == 1, int(tag) == 2) == tuple(
        bool(int(v)) for v in (boolean, false, true, bit, quantum)
    )
    assert bool(cast_operand_word(raw, domain, OperandKind.BOOLEAN)) == bool(int(truth))
    assert cast_operand_word(raw, domain, OperandKind.CLASSICAL).physical_word == int(
        bcast
    )
    assert cast_operand_word(raw, domain, OperandKind.QUANTUM).physical_word == int(
        qcast
    )
    assert int(equal) == 1


@pytest.mark.parametrize("row", fields("x"))
def test_complete_x_union_raw_projections(row: tuple[str, ...]) -> None:
    (
        word,
        payload,
        tag,
        text,
        streamed,
        boolean,
        bit,
        quantum,
        plus,
        minus,
        bcast,
        qcast,
        equal,
    ) = row
    raw = int(word)
    domain = OperandDomain.X_OPERAND
    assert (raw >> 29, raw & PAYLOAD_MASK) == (int(tag), int(payload))
    assert (
        format_operand_word(raw, domain)
        == native_diagnostic(text)
        == native_diagnostic(streamed)
    )
    assert (
        int(tag) == 3,
        int(tag) == 4,
        int(tag) == 2,
        raw == 3 << 29,
        raw == (3 << 29) | 1,
    ) == tuple(bool(int(v)) for v in (boolean, bit, quantum, plus, minus))
    assert cast_operand_word(raw, domain, OperandKind.X_CLASSICAL).physical_word == int(
        bcast
    )
    assert cast_operand_word(raw, domain, OperandKind.QUANTUM).physical_word == int(
        qcast
    )
    assert int(equal) == 1


@pytest.mark.parametrize("row", fields("mixed"))
def test_complete_mixed_union_raw_projections(row: tuple[str, ...]) -> None:
    (
        word,
        payload,
        tag,
        text,
        streamed,
        zlike,
        xlike,
        boolean,
        bit,
        quantum,
        xboolean,
        xbit,
        true,
        false,
        plus,
        minus,
        truth,
        xcast,
        bcast,
        qcast,
        xbcast,
        zcast,
        full_xcast,
        equal,
    ) = row
    raw = int(word)
    domain = OperandDomain.MIXED_OPERAND
    tag_value = int(tag)
    assert (raw >> 29, raw & PAYLOAD_MASK) == (tag_value, int(payload))
    assert (
        format_operand_word(raw, domain)
        == native_diagnostic(text)
        == native_diagnostic(streamed)
    )
    assert (
        tag_value in (0, 1, 2),
        tag_value in (2, 3, 4),
        tag_value == 0,
        tag_value == 1,
        tag_value == 2,
        tag_value == 3,
        tag_value == 4,
        raw == 1,
        raw == 0,
        raw == 3 << 29,
        raw == (3 << 29) | 1,
    ) == tuple(
        bool(int(v))
        for v in (
            zlike,
            xlike,
            boolean,
            bit,
            quantum,
            xboolean,
            xbit,
            true,
            false,
            plus,
            minus,
        )
    )
    assert bool(cast_operand_word(raw, domain, OperandKind.BOOLEAN)) == bool(int(truth))
    for kind, expected in (
        (OperandKind.X_BOOLEAN, xcast),
        (OperandKind.CLASSICAL, bcast),
        (OperandKind.QUANTUM, qcast),
        (OperandKind.X_CLASSICAL, xbcast),
    ):
        assert cast_operand_word(raw, domain, kind).physical_word == int(expected)
    assert raw == int(zcast) == int(full_xcast)
    assert int(equal) == 1


@pytest.mark.parametrize("row", fields("rawrestricted"))
def test_mutated_source_raw_words_remain_observable(row: tuple[str, ...]) -> None:
    word, btext, bftext, qftext, qmtext, qbtext, true, quantum, payload, qcast = row
    raw = int(word)
    for domain, text in (
        (OperandDomain.CLASSICAL_BOOLEAN, btext),
        (OperandDomain.CLASSICAL_FALSE, bftext),
        (OperandDomain.QUANTUM_FALSE, qftext),
        (OperandDomain.QUANTUM_MINUS, qmtext),
        (OperandDomain.QUANTUM_CLASSICAL, qbtext),
    ):
        assert format_operand_word(raw, domain) == native_diagnostic(text)
    assert (raw == 1, raw >> 29 == 2) == (bool(int(true)), bool(int(quantum)))
    assert raw & PAYLOAD_MASK == int(payload)
    assert cast_operand_word(
        raw, OperandDomain.QUANTUM_TRUE, OperandKind.QUANTUM
    ).physical_word == int(qcast)


@pytest.mark.parametrize("row", (*fields("order"), *fields("zorder")))
def test_all_canonical_equality_and_order(row: tuple[str, ...]) -> None:
    first, second, equal, less = row
    left, right = identity(int(first)), identity(int(second))
    assert (left == right) == bool(int(equal))
    assert (left < right) == bool(int(less))


@pytest.mark.parametrize("row", fields("zequals"))
def test_original_z_equality_overloads(row: tuple[str, ...]) -> None:
    word, *expected = row
    values = (
        (OperandKind.BOOLEAN, 0),
        (OperandKind.BOOLEAN, 1),
        (OperandKind.CLASSICAL, 0),
        (OperandKind.CLASSICAL, 17),
        (OperandKind.QUANTUM, 0),
        (OperandKind.QUANTUM, 17),
        (OperandKind.BOOLEAN, 0),
        (OperandKind.BOOLEAN, 1),
        (OperandKind.CLASSICAL, 0),
        (OperandKind.CLASSICAL, 17),
    )
    for (kind, value), equal in zip(values, expected, strict=True):
        assert (identity(int(word)) == OperandIdentity(kind, value)) == bool(int(equal))


@pytest.mark.parametrize("row", fields("xequals"))
def test_original_x_quantum_equality(row: tuple[str, ...]) -> None:
    assert (identity(int(row[0])) == OperandIdentity(OperandKind.QUANTUM, 0)) == bool(
        int(row[1])
    )


def test_complete_source_default_sentinels_and_literal_conversions() -> None:
    row = fields("defaults")[0]
    domains = (
        OperandDomain.CLASSICAL_BOOLEAN,
        OperandDomain.CLASSICAL_FALSE,
        OperandDomain.QUANTUM_FALSE,
        OperandDomain.Z_OPERAND,
        OperandDomain.X_OPERAND,
        OperandDomain.MIXED_OPERAND,
    )
    defaults = tuple(
        default_operand_identity(domain).physical_word for domain in domains
    )
    assert defaults == tuple(int(row[k]) for k in (0, 1, 6, 15, 16, 17))
    assert int(row[2]) == int(row[7]) == 0
    assert cast_operand_word(
        defaults[1], domains[1], OperandKind.CLASSICAL
    ).physical_word == int(row[3])
    assert format_operand_word(defaults[1], domains[1]) == row[4]
    assert admit_operand_identity(
        OperandIdentity(OperandKind.BOOLEAN, 0), domains[1]
    ).physical_word == int(row[5])
    assert cast_operand_word(
        defaults[2], domains[2], OperandKind.QUANTUM
    ).physical_word == int(row[8])
    assert format_operand_word(defaults[2], domains[2]) == row[9]
    with pytest.raises(ValueError, match="family"):
        admit_operand_identity(default_operand_identity(domains[2]), domains[2])
    for domain, kind, word, predicate in (
        (OperandDomain.QUANTUM_TRUE, OperandKind.BOOLEAN, row[10], row[11]),
        (OperandDomain.QUANTUM_MINUS, OperandKind.X_BOOLEAN, row[12], row[13]),
    ):
        item = OperandIdentity(kind, 1)
        assert admit_operand_identity(item, domain).physical_word == int(word)
        assert int(predicate) == 1
    assert format_operand_word(int(row[12]), OperandDomain.QUANTUM_MINUS) == row[14]
    # Only the false-restricted C++ wrapper permits numeric zero via consteval bool conversion.
    assert fields("numericfalse") == (("0",),)


_REFUSAL_DOMAINS = {
    "BitOrBool": OperandDomain.CLASSICAL_BOOLEAN,
    "QubitOrBit": OperandDomain.QUANTUM_CLASSICAL,
    "QubitOrFalse": OperandDomain.QUANTUM_FALSE,
    "QubitOrTrue": OperandDomain.QUANTUM_TRUE,
    "QubitOrMinusState": OperandDomain.QUANTUM_MINUS,
    "QubitOrBitOrBool": OperandDomain.Z_OPERAND,
    "QubitOrXBitOrXBool": OperandDomain.X_OPERAND,
    "QubitOrXZBitOrXZBool": OperandDomain.MIXED_OPERAND,
}


@pytest.mark.parametrize("row,numeric", [(row, 1) for row in _DATA["compile_refusals"]])
def test_all_original_compiler_refusals_have_native_admission_failures(
    row: dict[str, str], numeric: OperandIdentity
) -> None:
    case = row["case"]
    if case.endswith("no_default"):
        domain = (
            OperandDomain.QUANTUM_TRUE
            if case.startswith("true")
            else OperandDomain.QUANTUM_MINUS
        )
        with pytest.raises(ValueError, match="no default"):
            default_operand_identity(domain)
    elif case.endswith("restricted"):
        domain, kind, value = {
            "false_restricted": (OperandDomain.CLASSICAL_FALSE, OperandKind.BOOLEAN, 1),
            "true_restricted": (OperandDomain.QUANTUM_TRUE, OperandKind.BOOLEAN, 0),
            "minus_restricted": (OperandDomain.QUANTUM_MINUS, OperandKind.X_BOOLEAN, 0),
        }[case]
        with pytest.raises(ValueError, match="constant"):
            admit_operand_identity(OperandIdentity(kind, value), domain)
    else:
        # Parametrized helper supplies an untyped caller input to the typed boundary.
        with pytest.raises(ValueError, match="typed identity"):
            admit_operand_identity(numeric, _REFUSAL_DOMAINS[case])


@pytest.mark.parametrize("word", [-1, 1 << 32, True, 0.0, "0", None])
def test_raw_projection_validates_original_words(word: int) -> None:
    with pytest.raises(ValueError, match="unsigned uint32"):
        format_operand_word(word, OperandDomain.MIXED_OPERAND)
    with pytest.raises(ValueError, match="unsigned uint32"):
        cast_operand_word(word, OperandDomain.MIXED_OPERAND, OperandKind.BOOLEAN)


@pytest.mark.parametrize("domain", ["mixed_operand", 0, None])
def test_contexts_require_typed_domain(domain: OperandDomain) -> None:
    item = OperandIdentity(OperandKind.QUANTUM, 0)
    with pytest.raises(ValueError, match="OperandDomain"):
        admit_operand_identity(item, domain)
    with pytest.raises(ValueError, match="OperandDomain"):
        default_operand_identity(domain)
    with pytest.raises(ValueError, match="OperandDomain"):
        cast_operand_word(0, domain, OperandKind.QUANTUM)
    with pytest.raises(ValueError, match="OperandDomain"):
        format_operand_word(0, domain)


@pytest.mark.parametrize("domain", list(OperandDomain))
def test_unavailable_casts_keep_the_original_source_method_scope(
    domain: OperandDomain,
) -> None:
    with pytest.raises(ValueError, match="unavailable"):
        cast_operand_word(0, domain, OperandKind.REGISTER)
    if domain is OperandDomain.QUANTUM_TRUE:
        with pytest.raises(ValueError, match="no source diagnostic"):
            format_operand_word(1, domain)


@pytest.mark.parametrize("case", _DATA["gate_consumers"])
def test_union_consumers_match_original_gates_complex_states_and_lane_counts(
    case: GateConsumer,
) -> None:
    controls = {
        "false": OperandIdentity(OperandKind.BOOLEAN, 0),
        "true": OperandIdentity(OperandKind.BOOLEAN, 1),
        "classical": OperandIdentity(OperandKind.CLASSICAL, 1),
        "classical_alias": OperandIdentity(OperandKind.CLASSICAL, 0),
        "quantum": OperandIdentity(OperandKind.QUANTUM, 0),
    }
    targets = {
        "quantum": OperandIdentity(OperandKind.QUANTUM, 1),
        "x_classical": OperandIdentity(OperandKind.X_CLASSICAL, 0),
        "plus": OperandIdentity(OperandKind.X_BOOLEAN, 0),
        "minus": OperandIdentity(OperandKind.X_BOOLEAN, 1),
    }
    control = admit_operand_identity(controls[case["control"]], OperandDomain.Z_OPERAND)
    target = admit_operand_identity(targets[case["target"]], OperandDomain.X_OPERAND)
    assert identify_operand(bind_operand(control)) == control
    assert identify_operand(bind_operand(target)) == target
    gates = broadcast_cx_ir(control, target)
    original = read_kmx_circuit(case["kmx"])
    amplitudes = torch.tensor([1 + 2j, 3 - 4j, -2 + 5j, 7 + 1j], dtype=torch.complex128)
    amplitudes /= amplitudes.norm()
    for basis in case["basis_cases"]:
        first, second = basis["classical_bits"]
        prefix = tuple(
            ClassicalXor((), index, constant=bool(value)).instruction()
            for index, value in enumerate((first, second))
        )
        circuit = DynamicCircuit(2, inputs=amplitudes, dtype=torch.complex128)
        circuit.append(FieldOperation(2, (*prefix, *gates)))
        program = lower_reversible_phase(circuit.to_ir())
        text = encode_kmx(program)
        assert decode_kmx(text) == program
        records = tuple(
            record for record in read_kmx_circuit(text) if record.opcode != "INFO_TAG"
        )
        body = records[2 + first + second :]
        alias_scope = (
            case["control"] == "classical_alias" and case["target"] == "x_classical"
        )
        if alias_scope:
            assert [record.opcode for record in original] == [
                "PUSH_CONDITION",
                "NEG",
                "POP_CONDITION",
            ]
            assert [record.opcode for record in body] == ["NEG"]
            assert normalized_stream(
                KmxCircuit((), body).to_kmx()
            ) == normalized_stream(case["kmx"])
        else:
            assert body == tuple(original)
            assert PhysicalOperationStatistics.from_circuit(
                original
            ) == PhysicalOperationStatistics.from_circuit(KmxCircuit((), body))
        expected = torch.zeros(4, dtype=torch.complex128)
        quantum_words = basis["quantum_words"]
        for lane, phase in enumerate(basis["phase_half_turns"]):
            output = 2 * ((quantum_words[0] >> lane) & 1) + (
                (quantum_words[1] >> lane) & 1
            )
            expected[output] = amplitudes[lane] * exp(1j * pi * float(Fraction(phase)))
        result = run_dynamic(circuit, shots=2, seed=13, strategy="batched")
        torch.testing.assert_close(
            result.final_states.reshape(2, -1),
            expected.expand(2, -1),
            rtol=0,
            atol=1e-12,
        )
        assert result.final_states.numel() == 8  # Two logical qubits, zero work qubits.
        assert torch.all(result.classical_bits[:, 0] == bool(first))
        assert torch.all(result.classical_bits[:, 1] == bool(second))
        state = PackedBasisState(
            defaultdict(int, {0: 12, 1: 10}),
            defaultdict(int, {0: 15 * first, 1: 15 * second}),
            [0] * 4,
        )
        tally = PhysicalOperationCounters(4)
        apply_packed_basis_records(body, state, lambda: 0, observer=tally.observe)
        assert [state.qubits[k] for k in range(2)] == quantum_words
        assert [state.bits[k] for k in range(2)] == [15 * first, 15 * second]
        for lane, phase in enumerate(basis["phase_half_turns"]):
            exact = (state.phases[lane] + ((state.sign >> lane) & 1) * (1 << 127)) % (
                1 << 128
            )
            assert exact == int(Fraction(phase) * (1 << 127)) % (1 << 128)
            operations, levels = tally.snapshot(lane)
            counts = PhysicalOperationStatistics.from_counts(operations, levels)
            source_counts = basis["lane_counts"][lane]
            if alias_scope:
                assert (
                    source_counts.get("PUSH_CONDITION")
                    == source_counts.get("POP_CONDITION")
                    == 1
                )
                source_counts = {
                    name: count
                    for name, count in source_counts.items()
                    if name not in {"PUSH_CONDITION", "POP_CONDITION"}
                }
            assert dict(counts.items()) == source_counts
        # Static raw records include condition metadata independently of measured lanes.
        assert Counter(record.opcode for record in body) == Counter(
            record.opcode
            for record in original
            if record.opcode not in {"PUSH_CONDITION", "POP_CONDITION"}
            or not alias_scope
        )
