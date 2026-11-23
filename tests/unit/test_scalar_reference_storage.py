"""Keep frozen oracle payloads byte-identical under bounded XZ storage."""

import hashlib
import lzma
from pathlib import Path

import pytest

from tests.unit.fixtures.fixed_width_reference import read_cases as integer_cases
from tests.unit.fixtures.modular_integer_reference import read_cases as modular_cases

pytestmark = pytest.mark.unit
DATA = Path(__file__).with_name("data")


@pytest.mark.parametrize(
    "name,size,digest",
    [
        (
            "fixed_width_integer_reference.jsonl",
            7870040,
            "2e996c0a187e93f78854701977f366c5707f8b609fd48d2d46e9d3088cc6a8f6",
        ),
        (
            "modular_integer.jsonl",
            2082925,
            "f547bd20b7809e7dece1c540063300af91376d5e128fb859a3e6d39f1f548fa8",
        ),
        (
            "packed_chacha.jsonl",
            8536563,
            "2814d7f08f82f027dd69b56e59a8b5e16de0ff7c4862482ce2ca86fb98ad5e68",
        ),
        (
            "packed_word.json",
            2710938,
            "215afce20d8ad169d6279e61b7caf16d9df608c3bae884521eac63e5fec76f0e",
        ),
        (
            "population_counter.jsonl",
            2965477,
            "8eadf931688a401bfbc1aa7f3030036e54bee7126aaad8beb275b5c0ff57a8cf",
        ),
    ],
)
def test_frozen_decompressed_oracle_bytes(name: str, size: int, digest: str) -> None:
    actual = hashlib.sha256()
    count = 0
    with lzma.open(DATA / (name + ".xz"), "rb") as stream:
        while chunk := stream.read(65536):
            actual.update(chunk)
            count += len(chunk)
    assert count == size
    assert actual.hexdigest() == digest


def test_integer_generator_retains_plain_corpus_input(tmp_path: Path) -> None:
    packed = DATA / "fixed_width_integer_reference.jsonl.xz"
    plain = tmp_path / "integer.jsonl"
    with lzma.open(packed, "rb") as stream:
        plain.write_bytes(stream.readline())
    assert next(integer_cases(plain)) == next(integer_cases(packed))


def test_modular_generator_retains_plain_corpus_input(tmp_path: Path) -> None:
    packed = DATA / "modular_integer.jsonl.xz"
    plain = tmp_path / "modular.jsonl"
    with lzma.open(packed, "rb") as stream:
        plain.write_bytes(stream.readline())
    assert next(modular_cases(plain)) == next(modular_cases(packed))
