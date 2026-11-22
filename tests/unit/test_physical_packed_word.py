"""Compare complete packed lane operations with unchanged portable and SSE code."""

import copy
import json
from collections.abc import Callable, Iterable
from itertools import islice
from pathlib import Path

import pytest

from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from flagquantum.simulation._packed_chacha import PackedChaCha
from flagquantum.simulation._packed_word import (
    add_lanes,
    broadcast_u32,
    equal_u32,
    format_word,
    pack_lanes,
    random_word,
    set_word_bit,
    shift_lanes,
    unpack_lanes,
    word_mask,
)
from flagquantum.simulation.physical_basis_rng import _mt19937_64, packed_words
from flagquantum.simulation.physical_basis_shots import (
    BasisRecord,
    PackedBasisState,
    apply_packed_basis_records,
)
from flagquantum.simulation.physical_basis_spin import spin_basis
from flagquantum.simulation.physical_operation_counters import PhysicalOperationCounters
from tests.unit.fixtures.generate_operand_arrays import ArrayConsumer
from tests.unit.test_operand_array_views import (
    test_transformed_array_append_matches_original_execution as verify_array_execution,
)

pytestmark = pytest.mark.unit
_DATA = json.loads((Path(__file__).with_name("data") / "packed_word.json").read_text())


def test_complete_original_scalar_operations() -> None:
    for architecture, rows in _DATA["source_rows"].items():
        for row in rows:
            fields = row.split("\t")
            if not fields[0].startswith("b"):
                continue
            variant, width_text, _, operation, argument, a_text, b_text, expected = (
                fields
            )
            width, amount = int(width_text), int(argument)
            a, b, mask = int(a_text, 16), int(b_text, 16), word_mask(width)
            if operation in ("xor", "ixor"):
                actual = a ^ b
            elif operation in ("and", "iand"):
                actual = a & b
            elif operation in ("or", "ior"):
                actual = a | b
            elif operation == "not":
                actual = ~a & mask
            elif operation in ("add64", "iadd64", "add32"):
                actual = add_lanes(a, b, width, 32 if operation == "add32" else 64)
            elif operation == "eq32":
                actual = equal_u32(a, b, width)
            elif operation == "broadcast":
                actual = broadcast_u32(amount, width)
            elif operation in ("zero", "max"):
                actual = mask if operation == "max" else 0
            elif operation.startswith(("right", "left")):
                lane_bits = int(operation[-2:])
                actual = shift_lanes(
                    a, amount, width, lane_bits, left=operation.startswith("left")
                )
            elif operation in ("clear_bit", "set_bit"):
                actual = set_word_bit(a, amount, operation == "set_bit", width)
            else:
                assert operation == "from_u64s"
                actual = pack_lanes(unpack_lanes(a, width, 64), 64)
            assert actual == int(expected, 16), (architecture, variant, operation, row)


def test_original_getters_diagnostics_rng_and_dispatch() -> None:
    for architecture, rows in _DATA["source_rows"].items():
        for row in rows:
            fields = row.split("\t")
            kind = fields[0]
            if kind == "input64":
                _, _, values, expected = fields
                assert pack_lanes(
                    (int(value) for value in values.split(",") if value), 64
                ) == int(expected, 16)
            elif kind == "lanes32":
                _, _, width, word, lanes = fields
                assert unpack_lanes(int(word, 16), int(width), 32) == tuple(
                    int(value) for value in lanes.split(",") if value
                )
            elif kind == "diagnostic":
                _, _, width, word, other, nonzero, self_equal, other_equal, text = (
                    fields
                )
                value = int(word, 16)
                assert bool(value) == bool(int(nonzero))
                assert self_equal == "1"
                assert (value == int(other, 16)) == bool(int(other_equal))
                assert format_word(value, int(width)) == text
            elif kind == "str":
                _, width, word, text = fields
                assert format_word(int(word, 16), int(width)) == text
            elif kind in ("random", "randomize"):
                _, variant, width, seed, expected, after = fields
                rng = _mt19937_64(int(seed))
                high_first = variant == "b128_sse"
                if kind == "randomize":
                    random_word(int(width), rng, high_lane_first=high_first)
                    next(rng)
                actual = random_word(int(width), rng, high_lane_first=high_first)
                assert actual == int(expected, 16), (architecture, variant, kind)
                assert next(rng) == int(after)
            elif kind == "aliases":
                assert fields[1:] == [
                    "b64_polyfill",
                    "b128_sse" if architecture == "x86_64" else "b128_polyfill",
                    "b256_polyfill",
                    "b512_polyfill",
                ]


@pytest.mark.parametrize("width", [64, 128, 256, 512])
def test_existing_rng_and_chacha_replace_packing_without_consumer_changes(
    width: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    def consumer() -> tuple[list[int], list[int]]:
        words = packed_words(width)
        initialized = PackedChaCha(width).randomize(words)
        return list(islice(initialized.draws(), 65)), list(islice(words, 4))

    actual = consumer()

    def original_pack(values: Iterable[int], lane_bits: int) -> int:
        return sum(value << (lane_bits * index) for index, value in enumerate(values))

    def original_broadcast(value: int, width: int) -> int:
        return value * sum(1 << offset for offset in range(0, width, 32))

    monkeypatch.setattr(
        "flagquantum.simulation.physical_basis_rng.pack_lanes", original_pack
    )
    monkeypatch.setattr(
        "flagquantum.simulation._packed_chacha.pack_lanes", original_pack
    )
    monkeypatch.setattr(
        "flagquantum.simulation._packed_chacha.broadcast_u32", original_broadcast
    )
    assert consumer() == actual


@pytest.mark.parametrize("width", [0, -32, 33, True])
def test_invalid_width_refuses(width: int) -> None:
    with pytest.raises(ValueError, match="width"):
        word_mask(width)


@pytest.mark.parametrize("word", [-1, 1 << 64, True])
def test_invalid_word_refuses(word: int) -> None:
    with pytest.raises(ValueError, match="unsigned width"):
        format_word(word, 64)


def test_lane_and_bit_boundaries() -> None:
    for bits in (32, 64):
        assert pack_lanes((), bits) == 0
        assert shift_lanes(1, bits, 64, bits) == 0
        assert shift_lanes(1, 1 << 32, 64, bits) == 1
        assert shift_lanes(1, (1 << 64) - 1, 64, bits, left=True) == 0
        for invalid in (-1, 1 << 64, True):
            with pytest.raises(ValueError, match="shift count"):
                shift_lanes(1, invalid, 64, bits)
        for invalid in (-1, 1 << bits, True):
            with pytest.raises(ValueError, match="unsigned lane"):
                pack_lanes((invalid,), bits)
    with pytest.raises(ValueError, match="32 or 64"):
        pack_lanes((1,), 8)
    with pytest.raises(ValueError, match="integral number"):
        unpack_lanes(0, 32, 64)
    with pytest.raises(ValueError, match="integral uint64"):
        random_word(32, iter((0,)))
    for index in (-1, 64, True):
        with pytest.raises(IndexError, match="index"):
            set_word_bit(0, index, True, 64)
    with pytest.raises(ValueError, match="Boolean"):
        set_word_bit(0, 0, 1, 64)  # type: ignore[arg-type]
    with pytest.raises(StopIteration):
        random_word(128, iter((0,)))


@pytest.mark.parametrize(
    "case",
    json.loads((Path(__file__).with_name("data") / "operand_arrays.json").read_text())[
        "gate_consumers"
    ],
)
def test_existing_normal_append_full_complex_and_actual_resources(
    case: ArrayConsumer,
) -> None:
    verify_array_execution(case)


@pytest.mark.parametrize("width", [64, 128, 256, 512])
def test_spin_entry_replacement_preserves_feedback_exact_phases_and_lane_counts(
    width: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    circuit = read_kmx_circuit(
        "HMR q0 b0\nCX q1 q2 if b0\nZ_POW q2 0.5\n"
        "HMR q3 b1\nPUSH_CONDITION if b1\nZ q1\nPOP_CONDITION"
    )

    def consumer() -> (
        tuple[int, list[tuple[PackedBasisState, PhysicalOperationCounters]]]
    ):
        observations: list[tuple[PackedBasisState, PhysicalOperationCounters]] = []

        def capture(
            records: Iterable[BasisRecord],
            state: PackedBasisState,
            draw: Callable[[], int],
        ) -> None:
            counters = PhysicalOperationCounters(width)
            apply_packed_basis_records(records, state, draw, observer=counters.observe)
            assert state.qubits[0] == state.qubits[3] == 0
            observations.append((copy.deepcopy(state), counters))

        with monkeypatch.context() as context:
            context.setattr(
                "flagquantum.simulation.physical_basis_spin.apply_packed_basis_records",
                capture,
            )
            result = spin_basis(circuit.records, 4, 2, shots=width * 2 + 1, width=width)
        return result, observations

    actual_hash, actual_states = consumer()

    def original_pack(values: Iterable[int], lane_bits: int) -> int:
        return sum(value << (lane_bits * index) for index, value in enumerate(values))

    def original_broadcast(value: int, width: int) -> int:
        return value * sum(1 << offset for offset in range(0, width, 32))

    monkeypatch.setattr(
        "flagquantum.simulation.physical_basis_rng.pack_lanes", original_pack
    )
    monkeypatch.setattr(
        "flagquantum.simulation._packed_chacha.pack_lanes", original_pack
    )
    monkeypatch.setattr(
        "flagquantum.simulation._packed_chacha.broadcast_u32", original_broadcast
    )
    expected_hash, expected_states = consumer()
    assert actual_hash == expected_hash
    assert len(actual_states) == len(expected_states) == 3
    for (actual, counts), (expected, expected_counts) in zip(
        actual_states, expected_states, strict=True
    ):
        assert actual == expected
        assert counts.snapshot() == expected_counts.snapshot()
        for lane in range(width):
            assert counts.snapshot(lane) == expected_counts.snapshot(lane)
