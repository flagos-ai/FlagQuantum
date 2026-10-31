"""Bounded local acquisition without changing the native parser's authority."""

import os

import pytest

from flagquantum.ecosystem import InteropConversionError
from flagquantum.ecosystem._qmod import source as boundary
from flagquantum.ecosystem._qmod.adapter import SUPPORTED_SOURCE_VERSION
from flagquantum.ecosystem._qmod.syntax import MAX_BYTES

pytestmark = pytest.mark.unit
SOURCE = "qfunc main(output q:qbit) { allocate(q); X(q); }"


def test_text_file_and_symlink_have_the_same_owned_semantics(tmp_path):
    path = tmp_path / "source.qmod"
    path.write_bytes(SOURCE.encode())
    link = tmp_path / "link.qmod"
    link.symlink_to(path)
    imported = [
        boundary.load_qmod(SOURCE),
        boundary.load_qmod(path),
        boundary.load_qmod(link),
    ]
    assert len({item.program.semantic_identity for item in imported}) == 1
    assert [item.source_name for item in imported] == ["<qmod>", str(path), str(link)]
    assert all(item.source_version == SUPPORTED_SOURCE_VERSION for item in imported)


def test_existing_filename_string_remains_text_and_never_opens(tmp_path, monkeypatch):
    path = tmp_path / "exists.qmod"
    path.write_text(SOURCE)
    monkeypatch.setattr(
        boundary.os, "open", lambda *a, **k: pytest.fail("implicit file I/O")
    )
    with pytest.raises(InteropConversionError):
        boundary.load_qmod(str(path))


def test_file_read_is_bounded_and_descriptor_is_closed(tmp_path, monkeypatch):
    path = tmp_path / "oversized.qmod"
    path.write_bytes(b" " * (MAX_BYTES + 10))
    real_open, real_fdopen = os.open, os.fdopen
    descriptors, sizes = [], []

    def opened(*args, **kwargs):
        descriptor = real_open(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    class Stream:
        def __init__(self, *args, **kwargs):
            self.stream = real_fdopen(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, size):
            sizes.append(size)
            return self.stream.read(size)

    monkeypatch.setattr(boundary.os, "open", opened)
    monkeypatch.setattr(boundary.os, "fdopen", Stream)
    with pytest.raises(InteropConversionError) as error:
        boundary.load_qmod(path)
    assert error.value.report.issues[0].code == "qmod.budget.source"
    assert sizes == [MAX_BYTES + 1]
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_source_byte_boundary_and_structured_errors(tmp_path):
    exact = SOURCE + " " * (MAX_BYTES - len(SOURCE))
    assert boundary.load_qmod(exact).program.name == "main"
    with pytest.raises(InteropConversionError) as error:
        boundary.load_qmod(exact + " ", source_name="budget.qmod")
    report = error.value.report
    assert (report.adapter, report.direction, report.framework_version) == (
        "qmod",
        "import",
        None,
    )
    assert report.issues[0].code == "qmod.budget.source"
    assert "budget.qmod:1:1" in report.issues[0].message


@pytest.mark.parametrize("source_kind", ["text", "file"])
def test_encoding_errors_do_not_escape_as_private_diagnostics(tmp_path, source_kind):
    if source_kind == "text":
        source = "\ud800"
    else:
        source = tmp_path / "invalid.qmod"
        source.write_bytes(b"\xff\xfe")
    with pytest.raises(InteropConversionError) as error:
        boundary.load_qmod(source)
    assert error.value.report.issues[0].code == "qmod.encoding"


def test_crlf_and_comments_preserve_source_coordinates(tmp_path):
    path = tmp_path / "coordinates.qmod"
    path.write_bytes(
        b"// comment\r\nqfunc main(output q:qbit) {\r\nallocate(q);\r\nq=missing;\r\n}"
    )
    with pytest.raises(InteropConversionError) as error:
        boundary.load_qmod(path)
    assert error.value.report.issues[0].code == "qmod.name"
    assert f"{path}:4:" in str(error.value)
    assert b"\r\n" in path.read_bytes()


@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_nonregular_sources_are_refused_without_reading(tmp_path, kind):
    path = tmp_path / f"{kind}.qmod"
    if kind == "directory":
        path.mkdir()
    else:
        if not hasattr(os, "mkfifo"):
            pytest.skip("POSIX FIFO source test")
        os.mkfifo(path)
    with pytest.raises(InteropConversionError) as error:
        boundary.load_qmod(path)
    assert error.value.report.issues[0].code == "qmod.file"


def test_unknown_version_refuses_before_file_open(tmp_path, monkeypatch):
    monkeypatch.setattr(
        boundary.os,
        "open",
        lambda *a, **k: pytest.fail("file opened before version refusal"),
    )
    with pytest.raises(InteropConversionError) as error:
        boundary.load_qmod(tmp_path / "absent.qmod", source_version="future")
    assert error.value.report.issues[0].code == "qmod.version"


def test_file_errors_preserve_existing_oserror(tmp_path):
    with pytest.raises(FileNotFoundError):
        boundary.load_qmod(tmp_path / "absent.qmod")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_name": ""},
        {"source_name": 3},
        {"source_version": ""},
        {"source_version": 3},
    ],
)
def test_context_types_are_checked(kwargs):
    with pytest.raises(TypeError):
        boundary.load_qmod(SOURCE, **kwargs)


def test_path_and_input_refusals(tmp_path):
    class BytesPath:
        def __fspath__(self):
            return b"bytes.qmod"

    for source in (BytesPath(), b"bytes", 3):
        with pytest.raises(TypeError):
            boundary.load_qmod(source)
    for path in (tmp_path / "source.fq", tmp_path / "source.QMOD"):
        with pytest.raises(ValueError, match="suffix"):
            boundary.load_qmod(path)
    with pytest.raises(ValueError, match="source_name"):
        boundary.load_qmod(tmp_path / "source.qmod", source_name="spoofed.qmod")
