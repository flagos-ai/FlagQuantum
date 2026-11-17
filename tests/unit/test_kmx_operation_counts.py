"""Replay locked operation histograms from exact physical records."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from flagquantum.compiler.formats.kmb import encode_kmb
from flagquantum.compiler.formats.kmx_circuit import read_circuit_file
from flagquantum.compiler.formats.kmx_operation_counts import (
    format_operation_counts,
    main,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_EXPECTED = json.loads((_FIXTURES / "kickmix_operation_counts.json").read_text())
_ALL = json.loads((_FIXTURES / "kickmix_circuit_ownership.json").read_text())[
    "canonical"
]
_INPUTS = {
    "empty": "",
    "all_ops": _ALL,
    "guarded_mix": 'INFO_TAG "meta"\nX q3 if b1\nSWAP q1 q2\nCCX q0 q1 q2\n'
    "Z_POW q0 0.5\nDEBUG_PRINT q9 b5 if b7",
}


@pytest.mark.parametrize("case", tuple(_INPUTS))
@pytest.mark.parametrize(
    "options", ["basic", "--groups", "--magic", "--groups --magic"]
)
def test_locked_operation_count_modes(case: str, options: str) -> None:
    text = _INPUTS[case]
    for data in (text.encode(), encode_kmb(text)):
        circuit = read_circuit_file(data)
        assert (
            format_operation_counts(
                circuit, groups="--groups" in options, magic="--magic" in options
            )
            == _EXPECTED[case][options]
        )


def _run(*flags: str, data: bytes = b"") -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "flagquantum.compiler.formats.kmx_operation_counts",
            *flags,
        ],
        input=data,
        capture_output=True,
        timeout=20,
    )


def test_original_file_grouped_output_and_binary_pipe(tmp_path: Path) -> None:
    src, dst = tmp_path / "all.kmx", tmp_path / "counts.txt"
    src.write_text(_ALL)
    output = _run("--groups", "--in", str(src), "--out", str(dst))
    assert (output.returncode, output.stdout, output.stderr) == (0, b"", b"")
    assert dst.read_bytes() == _EXPECTED["all_ops"]["--groups"].encode()
    piped = _run("--groups", "--magic=", data=encode_kmb(_ALL))
    assert (piped.returncode, piped.stdout, piped.stderr) == (0, b"8\n", b"")


def test_invalid_flags_and_binary_leave_output_file_intact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "corrupt.kmb"
    path.write_bytes(b"\xd7")
    out = tmp_path / "counts.txt"
    out.write_text("previous\n")
    assert main(["--groups=true", "--out", str(out)]) == 1
    assert main(["--unknown", "--out", str(out)]) == 1
    assert main(["--in", str(path), "--out", str(out)]) == 1
    errors = capsys.readouterr().err
    assert "Unrecognized command line argument --unknown" in errors
    assert "Invalid KMB magic" in errors
    assert out.read_text() == "previous\n"


def test_file_mode_preserves_quoted_comment_character() -> None:
    circuit = read_circuit_file(b'INFO_TAG "left#right" # comment\r\nX q0\r')
    assert [record.tag for record in circuit] == ["left#right", None]
    assert "                 INFO_TAG: 1\n" in format_operation_counts(circuit)
