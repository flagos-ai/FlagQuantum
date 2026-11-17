"""Compare raw sample tables and debug traces to the locked command."""

import csv
import json
import subprocess
import sys
from pathlib import Path
from random import Random
from typing import TypedDict

import pytest

from flagquantum.compiler.formats.kmb import encode_kmb
from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from flagquantum.ecosystem.kmx_sample import main, sample_registers
from flagquantum.simulation.physical_basis_shots import sample_basis_batch


class _Case(TypedDict):
    kmx: str
    argv: list[str]
    stdout: str
    stderr: str


_SOURCE: dict[str, _Case] = json.loads(
    (Path(__file__).parent / "fixtures" / "kickmix_sample_command.json").read_text()
)


@pytest.mark.parametrize(
    "name", tuple(name for name in _SOURCE if name != "measurement")
)
@pytest.mark.parametrize("binary", [False, True])
def test_sample_command_matches_locked_bytes(
    name: str, binary: bool, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = _SOURCE[name]
    source, out = tmp_path / "circuit", tmp_path / "samples"
    source.write_bytes(encode_kmb(case["kmx"]) if binary else case["kmx"].encode())
    assert main([*case["argv"], "--in", str(source), "--out", str(out)]) == 0
    assert out.read_bytes() == case["stdout"].encode()
    assert capsys.readouterr().err == case["stderr"]


def test_sample_kmb_pipe() -> None:
    case = _SOURCE["rotation_only"]
    run = subprocess.run(
        [sys.executable, "-m", "flagquantum.ecosystem.kmx_sample"],
        input=encode_kmb(case["kmx"]),
        capture_output=True,
        timeout=20,
    )
    assert (run.returncode, run.stdout, run.stderr) == (
        0,
        case["stdout"].encode(),
        case["stderr"].encode(),
    )


def _check_shot(
    before: tuple[int, ...], after: tuple[int, ...], negative: bool
) -> tuple[int, int]:
    original, old_bit, original_work = before
    measured = after[1]
    assert old_bit == 0
    assert original >> 1 == original_work
    assert after == ((original & 1) ^ measured, measured, 0)
    # HMR contributes the pre-clear value after X; NEG toggles the same outcome.
    assert negative == bool(measured & original_work)
    return (original, measured)


def test_source_and_native_rows_preserve_measurement_feedback() -> None:
    case = _SOURCE["measurement"]
    raw = list(csv.reader(case["stdout"].splitlines()))[1:]
    source_histories = {
        _check_shot(
            tuple(int(cell, 2) for cell in row[0:6:2]),
            tuple(int(cell, 2) for cell in row[1:6:2]),
            int(row[6]) == -1,
        )
        for row in raw
    }
    native = list(
        sample_registers(read_kmx_circuit(case["kmx"]), 65, init="r0=*", rng=Random(37))
    )
    native_histories = {
        _check_shot(row.before, row.after, row.negative) for row in native
    }
    assert len(raw) == len(native) == 65
    assert (
        source_histories
        == native_histories
        == {(original, measured) for original in range(4) for measured in range(2)}
    )


def test_sign_mask_and_exact_rotation_are_separate() -> None:
    circuit = read_kmx_circuit("X q0\nHMR q0 b0\nX q1\nZ_POW q1 0.5\n")
    masks = iter([0b10])
    batch = sample_basis_batch(
        circuit,
        shots=2,
        rng=Random(0),
        outcome_mask=lambda: next(masks),
        capture_before=True,
    )
    assert batch.before is not None and not batch.before.qubits
    assert batch.sign_mask == 0b10 and batch.phase_units == (1 << 126, 3 << 126)
    assert not batch.qubits[0]


@pytest.mark.parametrize(
    "args",
    [
        ["--shots", "-1"],
        ["--shots", str(1 << 32)],
        ["--init", "r11=1"],
        ["--unknown"],
    ],
)
def test_invalid_sample_preserves_output(tmp_path: Path, args: list[str]) -> None:
    source, out = tmp_path / "circuit", tmp_path / "samples"
    source.write_text(_SOURCE["original"]["kmx"])
    out.write_bytes(b"previous")
    assert main([*args, "--in", str(source), "--out", str(out)]) == 1
    assert out.read_bytes() == b"previous"
