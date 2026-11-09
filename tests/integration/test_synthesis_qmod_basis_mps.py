"""Original high-width MCX checked without statevector materialization."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler._construction.selection import select_register_program
from flagquantum.core.ir import Instruction
from flagquantum.ecosystem._qmod.adapter import QmodImport
from flagquantum.ecosystem._qmod.source import load_qmod
from flagquantum.simulation.mps.entrypoints import run_mps
from flagquantum.simulation.mps.state import MPSState
from tools._qmod_basis_mps import probe_basis_mps

pytestmark = pytest.mark.integration
_DTYPE = torch.complex128
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "classiq_qmod"


def _original() -> QmodImport:
    manifest = json.loads((_FIXTURES / "mcx_sources.json").read_text())
    [row] = manifest["fixtures"]
    source = _FIXTURES / row["file"]
    data = source.read_bytes()
    assert hashlib.sha256(data).hexdigest() == row["sha256"]
    assert (
        hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        == row["git_blob_sha1"]
    )
    license_bytes = (_FIXTURES / manifest["license_file"]).read_bytes()
    assert hashlib.sha256(license_bytes).hexdigest() == manifest["license_sha256"]
    return load_qmod(source, source_version=manifest["commit"])


def test_original_mcx_stays_exact_and_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    imported = _original()
    selection = select_register_program(
        imported.program, imported.functions, include_boolean=True
    )
    row = selection.selected
    assert row and row.target and row.arithmetic
    circuit = row.target.submitted
    controls = selection.registers["cntrl"]
    [target] = selection.registers["target"]
    assert len(controls) == 50

    def forbid_dense(*args: object, **kwargs: object) -> NoReturn:
        pytest.fail("exact basis profile materialized a dense state")

    monkeypatch.setattr(MPSState, "to_statevector", forbid_dense)
    assert probe_basis_mps(imported, 2048) == {
        "stage": "execution_smoke_passed",
        "execution_profile": "basis_mps",
        "qubits": 64,
        "operations": 169,
        "depth": 167,
        "workspace": 13,
        "state_bytes": 2048,
        "basis_state": "0" * 64,
    }
    assert probe_basis_mps(imported, 2047)["stage"] == "planning_refused"

    for enabled in ((), controls[:-1], controls):
        before = tuple(Instruction("x", (wire,)) for wire in enabled)
        trial = replace(circuit, instructions=(*before, *circuit.instructions))
        reference = fq.Circuit(circuit.n_wires, dtype=_DTYPE)
        for wire in enabled:
            reference.x(wire)
        if len(enabled) == len(controls):
            reference.x(target)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            actual = run_mps(trial, dtype=_DTYPE)
            expected = run_mps(reference)
        assert actual.max_bond == expected.max_bond == 1
        assert not actual.truncation_records
        # Product sites give exact equality over the full state, including the
        # relative and absolute phases and all 13 workspace bits.
        assert all(
            torch.equal(left, right)
            for left, right in zip(actual.tensors, expected.tensors, strict=True)
        )


def test_basis_profile_refuses_coherent_circuit() -> None:
    source = "qfunc main(output q:qbit) { allocate(q); H(q); }"
    assert probe_basis_mps(load_qmod(source), 4096)["stage"] == "planning_refused"
