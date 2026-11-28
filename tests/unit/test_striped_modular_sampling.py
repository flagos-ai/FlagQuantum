"""Compare complete source stripe ordering, workspace and random draw position."""

import json
from collections import Counter, defaultdict
from collections.abc import Iterator, MutableSequence
from pathlib import Path

import pytest

from flagquantum.compiler.formats.kmb import KmxRecord
from flagquantum.core._fixed_width_integer import FixedWidthInteger
from flagquantum.simulation.fixed_width_sampling import sample_striped_modulo
from flagquantum.simulation.physical_basis_rng import _mt19937_64
from flagquantum.simulation.physical_basis_shots import (
    PackedBasisState,
    apply_packed_basis_records,
)

pytestmark = pytest.mark.unit


def test_complete_unchanged_source_streams() -> None:
    count = 0
    fixture = Path(__file__).with_name("data") / "striped_modular_sampling.jsonl"
    with fixture.open() as stream:
        for line in stream:
            case, expected = json.loads(line)
            width, literal, output_size, buffer_size, seed, repeats = case
            draws = _mt19937_64(seed)
            if output_size != width or buffer_size != width:
                assert expected == "REJECTED"
                # The immutable native result constructs both spans at width.
                continue
            if expected == "REJECTED":
                with pytest.raises(ValueError, match="width"):
                    modulus = FixedWidthInteger.from_text(literal, width)
                    sample_striped_modulo(modulus, draws)
                continue
            modulus = FixedWidthInteger.from_text(literal, width)
            output: tuple[int, ...] = ()
            scratch: tuple[int, ...] = ()
            for _ in range(repeats):
                output, scratch = sample_striped_modulo(modulus, draws)
            expected_output, expected_scratch, next_word = expected.split("|")
            assert output == tuple(map(int, expected_output.split())), case
            assert scratch == tuple(map(int, expected_scratch.split())), case
            assert next(draws) == int(next_word), case
            assert len(output) == len(scratch) == width
            assert all(
                sum(((word >> lane) & 1) << bit for bit, word in enumerate(output))
                < modulus.value
                for lane in range(64)
            )
            count += 1
    assert count == 724


def test_original_small_modulus_distribution_and_packed_execution() -> None:
    modulus = FixedWidthInteger(4, 9)
    draws = _mt19937_64(1084)
    counts: Counter[int] = Counter()
    for _ in range(10000):
        output, _ = sample_striped_modulo(modulus, draws)
        counts.update(
            sum(((word >> lane) & 1) << bit for bit, word in enumerate(output))
            for lane in range(64)
        )
    assert set(counts) == set(range(9))
    assert all(64000 <= count <= 80000 for count in counts.values())
    # Consume the stripes through the existing physical instruction executor.
    state = PackedBasisState(
        defaultdict(int, enumerate(output)), defaultdict(int), [0] * 64
    )
    records = (KmxRecord("CX", (0, 1)), KmxRecord("CX", (0, 1)))
    apply_packed_basis_records(records, state, lambda: 0)
    assert tuple(state.qubits[i] for i in range(4)) == output
    assert state.phases == [0] * 64
    assert not state.bits


def test_full_batch_and_input_ownership() -> None:
    modulus = FixedWidthInteger(3, 7)
    draws = iter([0, 0, 0, 123])
    output, scratch = sample_striped_modulo(modulus, draws)
    assert output == scratch == (0, 0, 0)
    assert next(draws) == 123
    with pytest.raises(ValueError, match="64-bit"):
        sample_striped_modulo(modulus, iter([1 << 64]))
    with pytest.raises(StopIteration):
        sample_striped_modulo(modulus, iter([]))


def test_shared_span_source_mutations_and_draw_position() -> None:
    from flagquantum.simulation.fixed_width_sampling import _sample_striped_modulo_into

    fixture = Path(__file__).with_name("data") / "striped_sampling_aliases.jsonl"
    checked = 0
    invalid_source_lanes = 0
    for line in fixture.read_text().splitlines():
        case, expected = json.loads(line)
        width, value, seed, output_offset, scratch_offset, repeats = case
        modulus = FixedWidthInteger(width, value)
        words = [(1 << 64) - 1] * (max(output_offset, scratch_offset) + width + 2)
        draws = _mt19937_64(seed)
        if expected == "UNDEFINED_COPY":
            with pytest.raises(ValueError, match="memcpy overlaps"):
                _sample_striped_modulo_into(
                    modulus, draws, words, output_offset, scratch_offset
                )
            continue
        for _ in range(repeats):
            _sample_striped_modulo_into(
                modulus, draws, words, output_offset, scratch_offset
            )
        storage, next_word = expected.split("|")
        assert words == list(map(int, storage.split())), case
        assert next(draws) == int(next_word), case
        # Source-defined overlap has a numerical defect; preserve literal output.
        if case == [4, 9, 0, 2, 2, 1]:
            invalid_source_lanes = sum(
                sum(
                    ((words[output_offset + bit] >> lane) & 1) << bit
                    for bit in range(width)
                )
                >= value
                for lane in range(64)
            )
        checked += 1
    assert checked == 419
    assert invalid_source_lanes == 11


def test_shared_span_validation_and_existing_entry_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum.simulation.fixed_width_sampling as sampling

    original = sampling._sample_striped_modulo_into
    calls: list[tuple[int, int]] = []

    def observed(
        modulus: FixedWidthInteger,
        draws: Iterator[int],
        words: MutableSequence[int],
        output_offset: int,
        scratch_offset: int,
    ) -> None:
        calls.append((output_offset, scratch_offset))
        original(modulus, draws, words, output_offset, scratch_offset)

    monkeypatch.setattr(sampling, "_sample_striped_modulo_into", observed)
    expected = sampling.sample_striped_modulo(FixedWidthInteger(4, 9), _mt19937_64(0))
    assert calls == [(0, 4)]
    words = [123] * 8
    original(FixedWidthInteger(4, 9), _mt19937_64(0), words, 0, 4)
    assert expected == (tuple(words[:4]), tuple(words[4:]))
    for offsets in ((-1, 4), (0, 5), (True, 4)):
        unchanged = [123] * 8
        with pytest.raises(ValueError, match="span"):
            original(FixedWidthInteger(4, 9), iter(()), unchanged, *offsets)
        assert unchanged == [123] * 8


def test_overlapping_full_batch_refuses_undefined_source_copy() -> None:
    from flagquantum.simulation.fixed_width_sampling import _sample_striped_modulo_into

    for output_offset, scratch_offset in ((0, 0), (0, 1), (1, 0)):
        with pytest.raises(ValueError, match="memcpy overlaps"):
            _sample_striped_modulo_into(
                FixedWidthInteger(3, 7),
                iter([0, 0, 0]),
                [123] * 4,
                output_offset,
                scratch_offset,
            )
