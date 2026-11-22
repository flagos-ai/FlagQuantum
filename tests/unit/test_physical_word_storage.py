"""Verify unchanged physical memory methods and their format-storage replacement."""

import copy
import json
import sys
from pathlib import Path

import pytest

from flagquantum.compiler.artifacts._word_arena import WordArena
from flagquantum.compiler.artifacts._word_storage import (
    AlignedBytes,
    WordPointer,
    WordSpan,
    max_u32,
    memcpy_repeat,
)
from flagquantum.compiler.formats.kmx_builder import KmxCircuitBuilder
from flagquantum.compiler.formats.kmx_circuit import read_kmb_circuit, read_kmx_circuit
from tests.unit.fixtures.generate_operand_arrays import ArrayConsumer
from tests.unit.test_operand_array_views import (
    test_transformed_array_append_matches_original_execution as verify_array_execution,
)

pytestmark = pytest.mark.unit
_DATA = json.loads(
    (Path(__file__).with_name("data") / "memory_support.json").read_text()
)
_ROWS: tuple[str, ...] = tuple(_DATA["source_rows"])


def ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def allocated_words(count: int, *, signed: bool = False) -> WordPointer:
    pointer = WordPointer(AlignedBytes(count * 4, 32), 0, 1, 4, signed)
    for k in range(count):
        pointer[k] = k
    return pointer


def test_complete_source_pointer_arithmetic_equality_and_mutation() -> None:
    base = allocated_words(40, signed=True)
    for row in _ROWS:
        fields = row.split("\t")
        if fields[0] != "pointer":
            continue
        (
            _,
            stride,
            displacement,
            shifted,
            old,
            next_value,
            equal,
            left_added,
            advanced,
            reset,
        ) = fields
        pointer = WordPointer(base.block, 12 * 4, int(stride), 4, True)
        original = pointer + 0
        moved = pointer + int(displacement)
        previous = pointer.advance_after()
        assert previous[0] == int(old)
        assert pointer.advance() is pointer
        assert pointer[0] == int(next_value)
        assert moved[0] == int(shifted)
        different_stride = WordPointer(
            base.block, moved.byte_offset, int(stride) + 1, 4, True
        )
        assert (moved == different_stride) == bool(int(equal))
        assert (int(displacement) + previous)[0] == int(left_added)
        pointer -= 2
        pointer += int(displacement)
        assert pointer[0] == int(advanced)
        assert (pointer - int(displacement))[0] == int(reset)
        assert original.byte_offset == 48
    zero = WordPointer(base.block, 12, 0, 4, True)
    zero[-999] = -1
    expected = next(row.split("\t")[1:] for row in _ROWS if row.startswith("mutate\t"))
    assert [zero[1000], base[3]] == list(map(int, expected))
    assert WordPointer() == WordPointer()
    assert WordPointer() != object()


def test_complete_source_span_transforms_iota_casts_and_bounds() -> None:
    base = allocated_words(40, signed=True)
    for row in _ROWS:
        fields = row.split("\t")
        if fields[0] == "span":
            (
                _,
                offset,
                stride,
                count,
                operation,
                argument,
                final_offset,
                final_stride,
                final_count,
                values,
            ) = fields
            span = WordSpan(
                WordPointer(base.block, int(offset) * 4, int(stride), 4, True),
                int(count),
            )
            if operation == "reversed":
                span = span.reversed()
            elif operation == "subspan":
                span = span.subspan(int(argument), 2)
            elif operation != "identity":
                transforms = {
                    "skip": span.skip,
                    "keep": span.keep,
                    "skip_last": span.skip_last,
                    "keep_last": span.keep_last,
                }
                span = transforms[operation](int(argument))
            assert (
                span.pointer.byte_offset // 4,
                span.pointer.stride,
                len(span),
                list(span),
            ) == (
                int(final_offset),
                int(final_stride),
                int(final_count),
                ints(values),
            ), row
            for index in range(span.count):
                assert span.at(index) == span[index]
        elif fields[0] == "iota":
            _, stride, count, offset, final_stride, value = fields
            span = WordSpan(
                WordPointer(base.block, 48, int(stride), 4, True), int(count)
            )
            pointer = span.indexed_pointer(2, negated=True)
            assert (pointer.byte_offset // 4, pointer.stride, pointer[0]) == (
                int(offset),
                int(final_stride),
                int(value),
            )
        elif fields[0] == "bounds":
            _, stride, count, refused = fields
            span = WordSpan(
                WordPointer(base.block, 48, int(stride), 4, True), int(count)
            )
            assert refused == "11"
            for index in (-1, int(count)):
                with pytest.raises(IndexError):
                    span.at(index)
    pointer = allocated_words(3)
    for k, cast_value in enumerate((0x01020304, 0x05060708, 0x090A0B0C)):
        pointer[k] = cast_value
    span = WordSpan(pointer, 3).cast_words(1)
    cast = next(row.split("\t")[1:] for row in _ROWS if row.startswith("cast\t"))
    assert [str(span.pointer.stride), str(span.count), ",".join(map(str, span))] == cast
    assert span.pointer.address == pointer.address
    assert span.indexed_pointer(1)[0] == span[1]
    assert list(span.subspan(1)) == list(span)[1:]
    assert "contents={" in span.diagnostic_text


def test_complete_source_helper_growth_capacity_alignment_and_clear() -> None:
    for alignment in (1, 2, 4, 8, 16, 32, 64):
        arena = WordArena(1, alignment)
        for row in _ROWS:
            fields = row.split("\t")
            if fields[0] != "helper" or int(fields[1]) != alignment:
                continue
            _, _, length, used, capacity, frozen_used, frozen_chunks, aligned = fields
            pointer = arena.grab_writeable(int(length))
            for k in range(int(length)):
                pointer[k] = k % 100
            assert (
                arena._used,
                arena._capacity,
                sum(span.count for span in arena._frozen),
                len(arena._frozen),
            ) == (int(used), int(capacity), int(frozen_used), int(frozen_chunks)), row
            assert (pointer.address % alignment == 0) == bool(int(aligned)), row
        moved = arena.move()
        assert len(arena) == 0 and arena._current is None and arena._capacity == 0
        arena.move_from(moved)
        arena.move_from(arena)
        retained = arena._current
        arena.clear()
        expected = next(
            row.split("\t")[2:]
            for row in _ROWS
            if row.startswith(f"helper_clear\t{alignment}\t")
        )
        assert (arena._capacity, arena._used, len(arena._frozen)) == tuple(
            map(int, expected)
        )
        assert arena._current is retained


@pytest.mark.parametrize("word_bytes,alignment", [(1, 32), (2, 4), (4, 4), (8, 32)])
def test_complete_source_arena_append_reserve_rewind_copy_and_move(
    word_bytes: int, alignment: int
) -> None:
    arena = WordArena(word_bytes, alignment)
    saved: WordPointer | None = None
    copied: WordArena | None = None
    for row in _ROWS:
        fields = row.split("\t")
        if fields[0] != "arena" or list(map(int, fields[1:3])) != [
            word_bytes,
            alignment,
        ]:
            continue
        _, _, _, action, count, chunks, values = fields
        if action == "repeat_zero":
            arena.append_repeat(2, 0)
        elif action == "single70":
            for k in range(70):
                arena.append(k)
        elif action == "many":
            arena.extend((7, 8, 9))
        elif action == "strided":
            raw = WordArena(word_bytes, alignment)
            raw.extend((21, 22, 23, 24, 25))
            pointer = next(raw.iter_spans()).pointer + 4
            pointer.stride = -2
            arena.extend(WordSpan(pointer, 3))
        elif action == "reverse":
            arena.append_reversed((21, 22, 23, 24, 25))
        elif action == "repeat9":
            arena.append_repeat(6, 9)
        elif action == "rewind3":
            saved = arena.grab_writeable(100)
            for k in range(100):
                saved[k] = 99
            arena.rewind_tail(saved + 3)
        elif action == "reuse4":
            pointer = arena.grab_writeable(4)
            assert saved is not None and pointer == saved + 3
            for k in range(4):
                pointer[k] = 50 + k
            copied = WordArena(word_bytes, alignment)
            copied.extend(arena)
            copied.append_reversed(arena)
            flat = WordPointer(
                AlignedBytes(len(arena) * word_bytes, alignment), 0, 1, word_bytes
            )
            arena.copy_into(flat)
            source_flat = next(
                r.split("\t")[3]
                for r in _ROWS
                if r.startswith(f"flat\t{word_bytes}\t{alignment}\t")
            )
            assert list(WordSpan(flat, len(arena))) == ints(source_flat)
        elif action == "move_back":
            previous_count = len(arena)
            moved = arena.move()
            source_move = next(
                r.split("\t")[3:]
                for r in _ROWS
                if r.startswith(f"move\t{word_bytes}\t{alignment}\t")
            )
            assert [len(arena), len(moved)] == [int(source_move[0]), previous_count]
            arena.move_from(moved)
            arena.move_from(arena)
        elif action == "clear":
            arena.clear()
        elif action == "reuse_after_clear":
            arena.append(3)
        assert len(arena) == int(count) and list(arena) == ints(values), row
        assert [span.count for span in arena.iter_spans()] == ints(chunks), row
        assert list(reversed(arena)) == list(arena)[::-1]
        assert arena[:] == list(arena)
        if arena:
            assert arena[-1] == list(arena)[-1]
    assert copied is not None
    source_copy = next(
        r.split("\t")[3]
        for r in _ROWS
        if r.startswith(f"copied\t{word_bytes}\t{alignment}\t")
    )
    assert list(copied) == ints(source_copy)
    copy_chunks = next(
        r.split("\t")[3]
        for r in _ROWS
        if r.startswith(f"copy_chunks\t{word_bytes}\t{alignment}\t")
    )
    assert [span.count for span in copied.iter_spans()] == ints(copy_chunks)
    assert (
        next(
            r.split("\t")[3]
            for r in _ROWS
            if r.startswith(f"rewind_refusal\t{word_bytes}\t{alignment}\t")
        )
        == "1"
    )
    with pytest.raises(ValueError, match="outside"):
        arena.rewind_tail(WordPointer(AlignedBytes(16, 32), 0, 1, word_bytes))


def test_complete_source_byte_utilities_and_maximum() -> None:
    for row in _ROWS:
        fields = row.split("\t")
        if fields[0] == "repeat":
            _, length, repetitions, expected = fields
            source = bytes(ord("a") + k for k in range(int(length)))
            destination = memoryview(bytearray(len(source) * int(repetitions)))
            memcpy_repeat(destination, source, int(repetitions))
            assert destination.tobytes() == expected.encode()
        elif fields[0] == "maximum":
            _, count, expected = fields
            pointer = allocated_words(int(count))
            for k in range(int(count)):
                pointer[k] = k * k % 255
            assert max_u32(WordSpan(pointer, int(count))) == int(expected)
    assert (
        next(row for row in _ROWS if row.startswith("allocation_overflow"))
        == "allocation_overflow\t1"
    )
    with pytest.raises(MemoryError):
        AlignedBytes(sys.maxsize, 32)


def test_native_failure_and_lifetime_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    arena = WordArena()
    arena.extend(range(20))
    pointer = next(arena.iter_spans()).pointer
    previous = (list(arena), arena._current, arena._used, arena._capacity)

    def fail_allocation(count: int, alignment: int) -> AlignedBytes:
        raise MemoryError("Injected allocation failure")

    with monkeypatch.context() as context:
        context.setattr(
            "flagquantum.compiler.artifacts._word_arena.AlignedBytes", fail_allocation
        )
        with pytest.raises(MemoryError):
            arena.grab_writeable(10000)
    assert (list(arena), arena._current, arena._used, arena._capacity) == previous
    arena.clear()
    assert pointer[5] == 5  # Pointers retain native allocation lifetime explicitly.
    arena.extend((1, 2))
    assert pointer[0] == 1  # Reusing the retained current chunk preserves aliasing.
    arena.extend(arena)
    assert list(arena) == [1, 2, 1, 2]
    before = list(arena)
    with pytest.raises(ValueError, match="width"):
        arena.extend((1, -1, 2))
    assert list(arena) == before
    with pytest.raises(ValueError, match="width"):
        arena.append(-1)
    with pytest.raises(ValueError, match="identical"):
        arena.move_from(WordArena(2, 32))
    with pytest.raises(ValueError, match="equal word"):
        arena.copy_into(WordPointer(AlignedBytes(100, 32), 0, 1, 2))
    with pytest.raises(IndexError):
        arena.copy_into(WordPointer(AlignedBytes(1, 8), 0, 1, 4))
    with pytest.raises(IndexError):
        _ = arena[len(arena)]
    with pytest.raises(IndexError):
        _ = arena["bad"]  # type: ignore[call-overload]
    with pytest.raises(TypeError, match="move-only"):
        copy.copy(arena)
    with pytest.raises(TypeError, match="move-only"):
        copy.deepcopy(arena)


@pytest.mark.parametrize("word_bytes", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("signed", [False, True])
def test_scalar_widths_and_reinterpretation(word_bytes: int, signed: bool) -> None:
    arena = WordArena(word_bytes, 32, signed=signed)
    bits = 8 * word_bytes - int(signed)
    minimum = -(1 << bits) if signed else 0
    arena.extend((minimum, (1 << bits) - 1))
    assert arena[:] == [minimum, (1 << bits) - 1]
    pointer = next(arena.iter_spans()).pointer
    assert pointer.signed == signed
    assert (pointer + 1)[0] == arena[1]
    if signed:
        raw = pointer.cast_words(word_bytes)
        assert raw[0] == 1 << bits
    with pytest.raises(ValueError):
        pointer[0] = 1 << bits
    arena.append_repeat(minimum, 3)
    assert arena[-3:] == [minimum] * 3
    target = WordPointer(
        AlignedBytes(len(arena) * word_bytes, 32), 0, 1, word_bytes, signed
    )
    arena.copy_into(target)
    assert list(WordSpan(target, len(arena))) == list(arena)


@pytest.mark.parametrize("alignment", [0, -1, 3, 2048, True])
def test_invalid_alignment_refuses(alignment: int) -> None:
    with pytest.raises(ValueError, match="Alignment"):
        AlignedBytes(3, alignment)


def test_native_pointer_and_span_bounds() -> None:
    with pytest.raises(ValueError, match="Alignment"):
        WordArena(4, 0)
    with pytest.raises(ValueError, match="integral"):
        WordArena(16, 8)
    with pytest.raises(ValueError, match="width"):
        WordPointer(word_bytes=3)
    with pytest.raises(ValueError, match="integers"):
        WordPointer(byte_offset=1.2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="displacement"):
        _ = WordPointer() + 1.2  # type: ignore[operator]
    with pytest.raises(IndexError):
        _ = WordPointer()[0]
    span = WordSpan(allocated_words(5), 5)
    assert (
        span.diagnostic_text
        == f"WordSpan{{.ptr={span.pointer.address}, .stride=1, .count=5, contents={{0, 1, 2, 3, 4}}}}"
    )
    span[1] = 7
    assert span[1] == 7
    with pytest.raises(ValueError, match="nonnegative"):
        span.skip(-1)
    with pytest.raises(ValueError, match="nonnegative"):
        WordSpan(span.pointer, -1)
    with pytest.raises(ValueError, match="four-byte"):
        max_u32(span.cast_words(1))
    with pytest.raises(ValueError, match="writable"):
        memcpy_repeat(memoryview(b"aa"), b"a", 2)
    with pytest.raises(ValueError, match="sufficient"):
        memcpy_repeat(memoryview(bytearray(1)), b"aa", 2)
    with pytest.raises(ValueError, match="nonnegative"):
        memcpy_repeat(memoryview(bytearray()), b"a", -1)
    with pytest.raises(ValueError, match="nonnegative"):
        AlignedBytes(-1, 32)
    arena = WordArena()
    with pytest.raises(ValueError, match="outside"):
        arena.rewind_tail(WordPointer(byte_offset=4))
    arena.extend(())
    arena.copy_into(WordPointer())
    arena.append_reversed(())
    assert len(arena) == 0
    pointer = arena.grab_writeable(1)
    with pytest.raises(ValueError, match="outside"):
        arena.rewind_tail(WordPointer(pointer.block, 1))


@pytest.mark.parametrize("storage", [WordArena, list])
def test_builder_storage_replacement(
    storage: type[WordArena] | type[list[int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("flagquantum.compiler.formats.kmb.WordArena", storage)
    text = "\n".join("CX q0 q1" for _ in range(1000))
    builder = KmxCircuitBuilder()
    builder.append_kmx_text(text)
    snapshot = builder.freeze()
    assert snapshot == read_kmx_circuit(text)
    assert read_kmb_circuit(builder.to_kmb()) == snapshot
    copied = copy.copy(builder)
    copied.append(builder)
    copied.append_reversed(builder)
    assert tuple(copied.freeze()) == tuple(snapshot) * 3
    builder.clear()
    builder.append_kmx_text("X q2")
    assert builder.freeze().to_kmx() == "X q2"
    assert len(snapshot.records) == 1000
    assert len(copied.freeze().records) == 3000


@pytest.mark.parametrize("storage", [WordArena, list])
@pytest.mark.parametrize(
    "case",
    json.loads((Path(__file__).with_name("data") / "operand_arrays.json").read_text())[
        "gate_consumers"
    ],
)
def test_storage_replacement_preserves_original_complex_execution(
    storage: type[WordArena] | type[list[int]],
    case: ArrayConsumer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("flagquantum.compiler.formats.kmb.WordArena", storage)
    verify_array_execution(case)


def test_contiguous_copy_preserves_raw_signed_words() -> None:
    arena = WordArena(signed=True)
    arena.extend((-1, -2, 3))
    target = WordPointer(AlignedBytes(12, 32), 0, -7, 4)
    arena.copy_into(target)
    assert list(WordSpan(target.cast_words(4), 1)) == [(1 << 32) - 1]
    target.stride = 1
    assert list(WordSpan(target, 3)) == [(1 << 32) - 1, (1 << 32) - 2, 3]
    arena = WordArena()
    arena.extend(WordArena())
    arena.append_reversed(WordArena())
    arena.extend(WordSpan(WordPointer(), 0))
    arena.append_repeat(1, 0)
    assert not arena
