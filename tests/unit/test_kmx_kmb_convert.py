"""Reproduce the locked converter command's incrementer and empty round trips."""

import subprocess
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from flagquantum.compiler.formats import kmb_convert


def _convert(
    direction: str, *paths: str, data: bytes = b""
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "flagquantum.compiler.formats.kmb_convert",
            direction,
            *paths,
        ],
        input=data,
        capture_output=True,
        timeout=20,
    )


@pytest.mark.parametrize(
    "text",
    [
        b"",
        b"APPEND_TO_REGISTER q0 r0\nAPPEND_TO_REGISTER q1 r0\n"
        b"APPEND_TO_REGISTER q2 r0\nCCX q0 q1 q2\nCX q0 q1\nX q0\n",
    ],
)
def test_locked_cli_roundtrip(tmp_path: Path, text: bytes) -> None:
    source = tmp_path / "source.kmx"
    binary = tmp_path / "bridge.kmb"
    output = tmp_path / "roundtrip.kmx"
    source.write_bytes(text)
    for direction, left, right in (
        ("kmx2kmb", source, binary),
        ("kmb2kmx", binary, output),
    ):
        run = _convert(direction, "--in", str(left), "--out", str(right))
        assert run.returncode == 0, run.stderr.decode()
    assert output.read_bytes() == text


def test_binary_stdstreams_and_invalid_input_leave_existing_output(
    tmp_path: Path,
) -> None:
    raw = _convert("kmx2kmb", data=b"X q0\n")
    assert raw.returncode == 0, raw.stderr.decode()
    restored = _convert("kmb2kmx", data=raw.stdout)
    assert (restored.returncode, restored.stdout) == (0, b"X q0\n")
    path = tmp_path / "output.kmx"
    path.write_bytes(b"original")
    bad = _convert("kmb2kmx", "--out", str(path), data=raw.stdout[:-1])
    assert bad.returncode and b"Truncated" in bad.stderr
    assert path.read_bytes() == b"original"


@pytest.mark.parametrize(
    "short_write,diagnostic",
    [(True, "Incomplete write"), (False, "cannot flush output")],
)
def test_converter_reports_output_failure(
    short_write: bool,
    diagnostic: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = Mock(spec=BytesIO, wraps=BytesIO())
    if short_write:
        output.write.return_value = 1
    else:
        output.flush.side_effect = OSError("cannot flush output")
    monkeypatch.setattr(
        kmb_convert,
        "sys",
        SimpleNamespace(
            stdin=SimpleNamespace(buffer=BytesIO(b"X q0\n")),
            stdout=SimpleNamespace(buffer=output),
            stderr=sys.stderr,
        ),
    )
    assert kmb_convert.main(["kmx2kmb"]) == 1
    assert diagnostic in capsys.readouterr().err
