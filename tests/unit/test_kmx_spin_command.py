"""Verify the locked spin hash, packed recurrence and benchmark report."""

import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

import pytest

from flagquantum.compiler.formats.kmb import encode_kmb
from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from flagquantum.compiler.formats.kmx_spin_report import format_spin_report
from flagquantum.ecosystem.kmx_spin import main
from flagquantum.simulation.physical_basis_shots import (
    PackedBasisState,
    apply_packed_basis_records,
)


class _Case(TypedDict):
    name: str
    argv: list[str]
    fixed: list[str]


_SOURCE = json.loads(
    (Path(__file__).parent / "fixtures" / "kickmix_spin_command.json").read_text()
)
_INPUTS: dict[str, str] = _SOURCE["inputs"]
_CASES: list[_Case] = _SOURCE["cases"]


@pytest.mark.parametrize(
    "case", _CASES, ids=lambda case: f"{case['name']} {' '.join(case['argv'])}"
)
@pytest.mark.parametrize("binary", [False, True])
def test_spin_hash_and_record_budget(case: _Case, binary: bool, tmp_path: Path) -> None:
    kmx = _INPUTS[case["name"]]
    src, out = tmp_path / "input", tmp_path / "report"
    src.write_bytes(encode_kmb(kmx) if binary else kmx.encode())
    assert main([*case["argv"], "--in", str(src), "--out", str(out)]) == 0
    lines = out.read_text().splitlines()
    assert lines[:2] == case["fixed"]
    assert len(lines) == 4
    assert re.fullmatch(
        r" {14}time elapsed: +[0-9.]+(?:e[+-][0-9]+)? [pnum kMGT]s", lines[2]
    )
    assert re.fullmatch(r" *[0-9.]+(?:e[+-][0-9]+)? [pnum kMGT]ops/sec", lines[3])


def test_original_spin_pair_accepts_1024_shots(tmp_path: Path) -> None:
    src, out = tmp_path / "circuit", tmp_path / "report"
    for mode, kmx in (
        ("64", "#CCX q0 q1 q2\n#CX q0 q1\n#X q0\n"),
        ("256", _INPUTS["incrementer"]),
    ):
        src.write_text(kmx)
        assert (
            main(
                ["--mode", mode, "--shots", "1024", "--in", str(src), "--out", str(out)]
            )
            == 0
        )
        assert out.read_text().startswith("hash: ")


def test_binary_stdin_and_stdout() -> None:
    case = next(row for row in _CASES if row["name"] == "measured")
    run = subprocess.run(
        [sys.executable, "-m", "flagquantum.ecosystem.kmx_spin", *case["argv"]],
        input=encode_kmb(_INPUTS[case["name"]]),
        capture_output=True,
        timeout=20,
    )
    assert run.returncode == 0 and not run.stderr
    assert run.stdout.decode().splitlines()[:2] == case["fixed"]


def test_timing_and_rate_use_integer_microseconds_and_source_padding() -> None:
    assert format_spin_report(0b101, 3, 1300, 1000) == (
        "hash: 1_1\n"
        "total simulated operations: 1300\n"
        "              time elapsed: 1.0 ms\n"
        "1.3 Mops/sec\n"
    )
    assert format_spin_report(0, 1, 0, 0).splitlines()[-2:] == [
        "              time elapsed: 1.0 us",
        "0 pops/sec",
    ]


def test_packed_state_keeps_angles_sign_and_work_across_scans() -> None:
    circuit = read_kmx_circuit(
        "X q0\nHMR q0 b0\nPUSH_CONDITION if b0\nX q1\nPOP_CONDITION\nZ_POW q1 0.5\n"
    )
    packed = PackedBasisState(defaultdict(int), defaultdict(int), [0, 0])
    tape = iter((0b01, 0b10))
    apply_packed_basis_records(circuit, packed, tape.__next__)
    assert (packed.qubits[0], packed.qubits[1], packed.bits[0], packed.sign) == (
        0,
        1,
        1,
        1,
    )
    assert packed.phases == [1 << 126, 0]
    apply_packed_basis_records(circuit, packed, tape.__next__)
    assert (packed.qubits[0], packed.qubits[1], packed.bits[0], packed.sign) == (
        0,
        3,
        2,
        3,
    )
    assert packed.phases == [1 << 127, 1 << 126]


@pytest.mark.parametrize(
    "argv",
    [
        ["--mode", "16", "--shots", "1"],
        ["--mode", "64", "--shots", "-1"],
        ["--mode", "128", "--shots", str(1 << 63)],
        ["--mode", "64"],
        ["--shots", "0"],
    ],
)
def test_invalid_spin_preserves_output(argv: list[str], tmp_path: Path) -> None:
    src, out = tmp_path / "input", tmp_path / "report"
    src.write_text(_INPUTS["incrementer"])
    out.write_text("previous")
    assert main([*argv, "--in", str(src), "--out", str(out)]) == 1
    assert out.read_text() == "previous"
