"""Compare complete source array/view operations with native immutable storage."""

import json
from cmath import exp
from collections import Counter, defaultdict
from collections.abc import Callable
from fractions import Fraction
from math import pi
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.artifacts._array_views import (
    cast_operand_payloads,
    checked_operand_array,
    copy_operands,
    default_operand_view,
    default_owned_operands,
    format_operand_array,
    operand_array_is_all,
    operand_view,
    recompute_operand_view,
)
from flagquantum.compiler.artifacts._operand_tags import PAYLOAD_MASK
from flagquantum.compiler.artifacts._operand_views import OperandDomain
from flagquantum.compiler.artifacts.operand_array import OperandArray
from flagquantum.compiler.artifacts.operand_identity import OperandIdentity, OperandKind
from flagquantum.compiler.formats._physical_operation_statistics import (
    PhysicalOperationStatistics,
)
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.formats.kmx_circuit import KmxCircuit, read_kmx_circuit
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.primitives.broadcast_binding import broadcast_cx_ir
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.physical_basis_shots import (
    PackedBasisState,
    apply_packed_basis_records,
)
from flagquantum.simulation.physical_operation_counters import PhysicalOperationCounters
from tests.unit.fixtures.generate_operand_arrays import ArrayConsumer

pytestmark = pytest.mark.unit
_DATA = json.loads(
    (Path(__file__).with_name("data") / "operand_arrays.json").read_text()
)
_ROWS: tuple[str, ...] = tuple(_DATA["source_rows"])


def array_from_words(text: str) -> OperandArray:
    words = (int(word) for word in text.split(",") if word)
    return OperandArray(
        OperandIdentity(OperandKind(word >> 29), word & PAYLOAD_MASK) for word in words
    )


def word_text(array: OperandArray) -> str:
    return ",".join(str(item.physical_word) for item in array)


def native_text(text: str) -> str:
    return (
        text.replace("stride_span_xz{", "OperandView{")
        .replace("stride_span_z{", "OperandView{")
        .replace("array_xz{", "OperandArray{")
    )


def test_complete_source_view_corpus() -> None:
    # Iterate one corpus so exhaustive small arrays share fixture and import costs.
    targets = (
        OperandKind.QUANTUM,
        OperandKind.CLASSICAL,
        OperandKind.BOOLEAN,
        OperandKind.X_CLASSICAL,
        OperandKind.X_BOOLEAN,
        OperandDomain.X_OPERAND,
        OperandDomain.Z_OPERAND,
        OperandDomain.QUANTUM_CLASSICAL,
        OperandDomain.CLASSICAL_BOOLEAN,
    )
    for row in _ROWS:
        fields = row.split("\t")
        if fields[0] != "view":
            continue
        (
            _,
            domain,
            base,
            hint,
            operation,
            argument,
            offset,
            stride,
            count,
            type_hint,
            computed,
            flags,
            expected_words,
            diagnostic,
            checks,
            z_flags,
            z_computed,
        ) = fields
        original = operand_view(array_from_words(base), int(hint))
        view = original[::2] if domain.endswith("step2") else original
        if operation == "reversed":
            view = view.reversed()
        elif operation != "identity":
            transforms = {
                "skip": view.skip,
                "keep": view.keep,
                "skip_last": view.skip_last,
                "keep_last": view.keep_last,
            }
            view = transforms[operation](int(argument))
        assert word_text(view) == expected_words, row
        assert (view.stride, len(view), view.common_type) == (
            int(stride),
            int(count),
            int(type_hint),
        ), row
        if view:
            assert view.offset == int(offset), row
            assert view[0] == tuple(view)[0] and view[-1] == tuple(view)[-1]
        assert view._owner is original._owner
        assert recompute_operand_view(view).common_type == int(computed), row
        actual = [operand_array_is_all(view, target) for target in targets]
        # The classical X predicate shares the existing classical Z union with H.
        classical_x = operand_array_is_all(
            view, OperandDomain.X_OPERAND, classical_only=True
        )
        actual.insert(8, classical_x)
        assert "".join(str(int(value)) for value in actual) == flags, row
        checked_targets: tuple[OperandKind | OperandDomain, ...] = (
            OperandKind.QUANTUM,
            OperandKind.CLASSICAL,
            OperandKind.X_CLASSICAL,
            OperandDomain.QUANTUM_CLASSICAL,
            OperandDomain.Z_OPERAND,
            OperandDomain.X_OPERAND,
        )
        for target, succeeds in zip(checked_targets, checks, strict=True):
            if succeeds == "1":
                assert checked_operand_array(view, target, "probe") is view
            else:
                with pytest.raises(ValueError, match="probe contained"):
                    checked_operand_array(view, target, "probe")
        context = (
            OperandDomain.Z_OPERAND
            if domain.startswith("z")
            else OperandDomain.MIXED_OPERAND
        )
        assert format_operand_array(view, context) == native_text(diagnostic), row
        if z_flags != "-":
            native_classical = operand_array_is_all(
                view, OperandDomain.CLASSICAL_BOOLEAN
            )
            source_classical = native_classical and (
                not view or view.common_type in (0, 1, 128)
            )
            assert source_classical == bool(int(z_flags[0])), row
            # Preserve the upstream has_any_qubits always-true defect as evidence.
            assert z_flags[1] == "1"
            assert recompute_operand_view(view, z_only=True).common_type == int(
                z_computed
            ), row


def test_complete_source_owning_copy_concat_and_move_disposition() -> None:
    for row in _ROWS:
        fields = row.split("\t")
        if fields[0] == "owner":
            (
                _,
                base,
                reversed_words,
                hint,
                moved_hint,
                concat_words,
                concat_hint,
                transferred,
                self_move,
                diagnostic,
            ) = fields
            original = array_from_words(base)
            copied = copy_operands(original.reversed())
            assert word_text(copied) == reversed_words
            assert copied.common_type == int(hint) == int(moved_hint)
            assert copied.offset == 0 and copied.stride == 1
            assert copied._owner is not original._owner or not copied
            joined = original + copied
            assert word_text(joined) == concat_words
            assert joined.common_type == int(concat_hint)
            assert format_operand_array(
                copied, OperandDomain.MIXED_OPERAND, owned=True
            ) == native_text(diagnostic)
            assert transferred == self_move == "1"
            # Immutable native references share ownership without a moved-from state.
            retained = copied
            del copied
            assert word_text(retained) == reversed_words
        elif fields[0] == "zowner":
            _, base, hint, moved_hint, concat_words, concat_hint, transferred = fields
            original = copy_operands(array_from_words(base), z_only=True)
            assert original.common_type == int(hint) == int(moved_hint)
            joined = copy_operands(original.skip(1) + original.keep(1), z_only=True)
            assert word_text(joined) == concat_words
            assert joined.common_type == int(concat_hint)
            assert transferred == "1"


def test_defaults_repeat_and_lifetime() -> None:
    defaults = next(
        row.split("\t")[1:] for row in _ROWS if row.startswith("defaults\t")
    )
    assert defaults == ["0", "0", "2", "2", "2", "0", "0", "0"]
    assert default_owned_operands().common_type == 0
    default = default_operand_view()
    assert (len(default), default.stride, default.common_type) == (0, 0, 2)
    repeat = OperandArray.repeat_false(7)
    source = next(row.split("\t")[1:] for row in _ROWS if row.startswith("repeat\t"))
    assert source == [
        str(repeat.stride),
        str(len(repeat)),
        str(repeat.common_type),
        word_text(repeat),
    ]
    assert len(repeat._owner) == 1
    assert repeat[-1] == repeat[0]
    assert list(repeat[::-2]) == [repeat[0]] * 4
    assert list(repeat.reversed().skip(2).keep(3)) == [repeat[0]] * 3
    assert len(OperandArray.repeat_false(10**9)._owner) == 1
    assert not OperandArray.repeat_false(0)
    with pytest.raises(IndexError):
        _ = OperandArray.repeat_false(0)[0]
    view = array_from_words("1073741824,1073741825,1073741826").reversed().skip(1)
    assert [item.value for item in view] == [1, 0]
    assert view._owner[2].value == 2
    # Native construction snapshots mutable inputs and never exposes uninitialized cells.
    values = [OperandIdentity(OperandKind.QUANTUM, 0)]
    owned = OperandArray(values)
    values[0] = OperandIdentity(OperandKind.QUANTUM, 1)
    assert owned[0].value == 0
    for row in _ROWS:
        if row.startswith("allocation\t"):
            _, tag, count, result_tag, allocated = row.split("\t")
            assert tag == result_tag and count == "4" and allocated == "1"


@pytest.mark.parametrize("count", [-1, True, 1.2])
def test_view_count_refusals(count: int) -> None:
    array = array_from_words("1073741824")
    for transform in (
        OperandArray.repeat_false,
        array.skip,
        array.keep,
        array.skip_last,
        array.keep_last,
    ):
        with pytest.raises(ValueError, match="nonnegative integer"):
            transform(count)


def test_checked_and_unchecked_cast_boundaries() -> None:
    array = array_from_words(
        "0,1,536870912,536870913,1073741824,1073741825,1610612736,1610612737,2147483648,2147483649"
    )
    for kind in (OperandKind.QUANTUM, OperandKind.CLASSICAL, OperandKind.X_CLASSICAL):
        cast = cast_operand_payloads(array, kind)
        assert [item.value for item in cast] == [0, 1] * 5
        assert all(item.kind is kind for item in cast)
    assert [
        item.value for item in cast_operand_payloads(array, OperandKind.BOOLEAN)
    ] == [0, 1] * 5
    assert [
        item.value for item in cast_operand_payloads(array, OperandKind.X_BOOLEAN)
    ] == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    with pytest.raises(ValueError, match="family range"):
        cast_operand_payloads(array_from_words("1073741826"), OperandKind.BOOLEAN)
    with pytest.raises(ValueError, match="cast target"):
        cast_operand_payloads(array, OperandKind.REGISTER)
    with pytest.raises(ValueError, match="target family"):
        operand_array_is_all(array, OperandKind.REGISTER)
    with pytest.raises(ValueError, match="type hint"):
        operand_view(array, 5)
    with pytest.raises(ValueError, match="contradicts"):
        operand_view(array, 2)
    with pytest.raises(ValueError, match="outside"):
        copy_operands(array, z_only=True)
    with pytest.raises(ValueError, match="diagnostics"):
        format_operand_array(array, OperandDomain.X_OPERAND)
    with pytest.raises(ValueError, match="typed identities"):
        OperandArray([0])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="register identities"):
        OperandArray((OperandIdentity(OperandKind.REGISTER, 0),))
    assert OperandArray() != object()
    with pytest.raises(TypeError, match="Concatenation"):
        _ = array + object()  # type: ignore[operator]
    with pytest.raises(IndexError, match="integer or slice"):
        _ = array[1.5]  # type: ignore[call-overload]


def test_source_memory_cast_projections_and_x_view_conversions() -> None:
    for row in _ROWS:
        fields = row.split("\t")
        if fields[0] == "cast":
            _, words, expected, bool_stride, id_stride = fields
            array = array_from_words(words)
            kinds = (
                OperandKind.QUANTUM,
                OperandKind.CLASSICAL,
                OperandKind.X_CLASSICAL,
                OperandKind.BOOLEAN,
                OperandKind.X_BOOLEAN,
            )
            columns = [cast_operand_payloads(array, kind) for kind in kinds]
            actual = ",".join(
                ":".join(str(item.value) for item in lane)
                for lane in zip(*columns, strict=True)
            )
            assert actual == expected, row
            assert int(bool_stride) == 4 * array.stride
            assert int(id_stride) == array.stride
        elif fields[0] == "xview":
            _, words, hint, stride, output_words, aliases = fields
            array = operand_view(array_from_words(words), 64)
            assert (array.common_type, array.stride, word_text(array)) == (
                int(hint),
                int(stride),
                output_words,
            )
            assert aliases == "1"
    ownership = next(
        row.split("\t")[1:] for row in _ROWS if row.startswith("ownership\t")
    )
    assert ownership == ["1", "1", "2", "2", "2", "3", "4", "1", "1"]


@pytest.mark.parametrize("case", _DATA["gate_consumers"])
def test_transformed_array_append_matches_original_execution(
    case: ArrayConsumer,
) -> None:
    controls = array_from_words("1073741824,536870912,1,0")
    targets = OperandArray(OperandIdentity(OperandKind.QUANTUM, k) for k in range(1, 5))
    transforms: dict[str, Callable[[OperandArray], OperandArray]] = {
        "identity": lambda v: v,
        "reversed": OperandArray.reversed,
        "skip": lambda v: v.skip(1),
        "keep": lambda v: v.keep(2),
        "skip_last": lambda v: v.skip_last(1),
        "keep_last": lambda v: v.keep_last(2),
        "step2": lambda v: v[::2],
    }
    if case["operation"] == "repeat":
        controls = OperandArray.repeat_false(4)
    else:
        transform = transforms[case["operation"]]
        controls, targets = transform(controls), transform(targets)
    gates = broadcast_cx_ir(controls, targets)
    source = read_kmx_circuit(case["kmx"])
    amplitudes = torch.complex(
        torch.arange(1, 33, dtype=torch.float64),
        2 * torch.arange(32, dtype=torch.float64) - 9,
    )
    amplitudes /= amplitudes.norm()
    prefix = ClassicalXor((), 0, constant=case["enabled"]).instruction()
    circuit = DynamicCircuit(5, inputs=amplitudes, dtype=torch.complex128)
    circuit.append(FieldOperation(5, (prefix, *gates)))
    program = lower_reversible_phase(circuit.to_ir())
    text = encode_kmx(program)
    assert decode_kmx(text) == program
    body = tuple(r for r in read_kmx_circuit(text) if r.opcode != "INFO_TAG")[
        1 + int(case["enabled"]) :
    ]
    assert body == tuple(source)
    assert Counter(r.opcode for r in body) == Counter(r.opcode for r in source)
    expected = torch.empty_like(amplitudes)
    for lane, phase in enumerate(case["phase_half_turns"]):
        output = sum(
            ((word >> lane) & 1) << (4 - bit)
            for bit, word in enumerate(case["quantum_words"])
        )
        expected[output] = amplitudes[lane] * exp(1j * pi * float(Fraction(phase)))
    result = run_dynamic(circuit, shots=2, seed=11, strategy="batched")
    torch.testing.assert_close(
        result.final_states.reshape(2, -1), expected.expand(2, -1), rtol=0, atol=1e-12
    )
    assert result.final_states.numel() == 64  # Five logical qubits and zero workspace.
    assert torch.all(result.classical_bits[:, 0] == case["enabled"])
    initial = {
        bit: sum(((lane >> (4 - bit)) & 1) << lane for lane in range(32))
        for bit in range(5)
    }
    state = PackedBasisState(
        defaultdict(int, initial),
        defaultdict(int, {0: ((1 << 32) - 1) * case["enabled"]}),
        [0] * 32,
    )
    uncounted = PackedBasisState(
        defaultdict(int, initial), defaultdict(int, state.bits), [0] * 32
    )
    tally = PhysicalOperationCounters(32)
    apply_packed_basis_records(body, state, lambda: 0, observer=tally.observe)
    apply_packed_basis_records(body, uncounted, lambda: 0)
    assert state == uncounted
    assert [state.qubits[k] for k in range(5)] == case["quantum_words"]
    for lane, phase in enumerate(case["phase_half_turns"]):
        exact = (state.phases[lane] + ((state.sign >> lane) & 1) * (1 << 127)) % (
            1 << 128
        )
        assert exact == int(Fraction(phase) * (1 << 127)) % (1 << 128)
        assert (
            dict(PhysicalOperationStatistics.from_counts(*tally.snapshot(lane)).items())
            == case["lane_counts"][lane]
        )
    assert PhysicalOperationStatistics.from_circuit(
        KmxCircuit((), body)
    ) == PhysicalOperationStatistics.from_circuit(source)
