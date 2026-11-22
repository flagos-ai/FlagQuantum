"""Compare complete source stripe ordering, workspace and random draw position."""

import json
from collections import Counter, defaultdict
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
