"""Replace complete original test-support routes with scoped native ownership."""

import io
import json
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

import pytest

from flagquantum.simulation.physical_basis_rng import (
    _mt19937_64,
    _mt19937_64_seed_sequence,
)
from tests.unit.shared_test_fixtures import (
    IndependentTestRng,
    externally_seeded_rng,
    resolve_testdata_file_path,
    rewind_read_close,
    temporary_named_file,
)

pytestmark = pytest.mark.unit


def observations() -> dict[str, object]:
    data: object = json.loads(
        (Path(__file__).with_name("data") / "shared_test_support.json").read_text()
    )
    if not isinstance(data, dict) or not all(isinstance(key, str) for key in data):
        raise ValueError("Invalid shared test observations")
    return data


def test_complete_standard_seed_initialization() -> None:
    records = observations()["seed"]
    assert isinstance(records, list)
    for seeds, expected in records:
        assert list(islice(_mt19937_64_seed_sequence(tuple(seeds)), 630)) == expected


def test_original_independent_children_and_parent_draw_order() -> None:
    records = observations()["independent"]
    assert isinstance(records, list)
    for seed, expected in records:
        # Inject the original scalar parent to isolate its four-word child route.
        rng = IndependentTestRng(iter(()))
        parent = _mt19937_64(seed)
        rng._shared = parent
        actual: list[int] = []
        children = [rng.draw_stream() for _ in range(3)]
        for child in children:
            actual.extend(islice(child, 630))
        actual.append(next(parent))
        assert actual == expected


def test_entropy_ownership_is_lazy_once_and_child_seeds_are_eager() -> None:
    consumed = []

    def words() -> Iterator[int]:
        for value in range(8):
            consumed.append(value)
            yield value

    rng = IndependentTestRng(words())
    assert consumed == []
    first = rng.draw_stream()
    second = rng.draw_stream()
    assert consumed == list(range(8))
    parent = externally_seeded_rng(iter(range(8)))
    expected_first = _mt19937_64_seed_sequence(tuple(islice(parent, 4)))
    expected_second = _mt19937_64_seed_sequence(tuple(islice(parent, 4)))
    assert list(islice(second, 630)) == list(islice(expected_second, 630))
    assert list(islice(first, 630)) == list(islice(expected_first, 630))
    assert consumed == list(range(8))


@pytest.mark.parametrize("layout", range(64))
def test_complete_original_lookup_priority(tmp_path: Path, layout: int) -> None:
    records = observations()["lookup"]
    assert isinstance(records, list)
    directories, files, result, after = records[layout]
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    prefixes = (cwd / "kickmix/testdata", cwd / "testdata", tmp_path / "testdata")
    for index, prefix in enumerate(prefixes):
        if directories & (1 << index):
            prefix.mkdir(parents=True)
            if files & (1 << index):
                (prefix / "asset").write_bytes(bytes([index]))
    if result[0] == "error":
        with pytest.raises(ValueError) as failure:
            resolve_testdata_file_path("asset", cwd)
        assert str(failure.value).encode().hex() == result[1]
    else:
        assert resolve_testdata_file_path("asset", cwd).encode().hex() == result[0]
    assert [
        (prefix / "asset").read_bytes().hex() if (prefix / "asset").is_file() else None
        for prefix in prefixes
    ] == after


def test_lookup_c_string_and_directory_failures(tmp_path: Path) -> None:
    prefix = tmp_path / "testdata"
    prefix.mkdir()
    (prefix / "asset").write_bytes(b"kept")
    assert (
        resolve_testdata_file_path("asset\0ignored", tmp_path)
        == "testdata/asset\0ignored"
    )
    # Original fopen tests only acquisition, so readable directories also qualify.
    (prefix / "directory").mkdir()
    assert resolve_testdata_file_path("directory", tmp_path) == "testdata/directory"
    assert observations()["directory"] == [b"testdata/asset".hex()]
    assert (prefix / "asset").read_bytes() == b"kept"


def test_complete_file_bytes_and_original_descriptor_defect() -> None:
    records = observations()["files"]
    assert isinstance(records, list)
    contents = bytes(range(256))
    assert records[:2] == [contents.hex(), contents.hex()]
    assert records[2:4] == [b"replacement".hex()] * 2
    assert records[4:] == ["0 1", ""]
    with temporary_named_file(contents) as path:
        assert path.read_bytes() == contents
        path.write_bytes(b"replacement")
        stream = path.open("rb")
        stream.read(1)
        assert rewind_read_close(stream) == b"replacement"
        assert stream.closed
    assert not path.exists()
    with temporary_named_file() as empty:
        assert empty.read_bytes() == b""
    assert not empty.exists()


def test_temporary_cleanup_after_failure_and_removed_path() -> None:
    with (
        pytest.raises(RuntimeError, match="failure"),
        temporary_named_file(b"x") as path,
    ):
        raise RuntimeError("failure")
    assert not path.exists()
    with temporary_named_file() as removed:
        removed.unlink()
    assert not removed.exists()


def test_rewind_closes_after_read_error() -> None:
    class BrokenStream(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            raise OSError("read failure")

    stream = BrokenStream(b"x")
    with pytest.raises(OSError, match="read failure"):
        rewind_read_close(stream)
    assert stream.closed
