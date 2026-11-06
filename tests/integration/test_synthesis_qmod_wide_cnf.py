"""Verify the original 67-variable phase oracle and coherent clause semantics."""

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._construction.selection import select_register_program
from flagquantum.core.ir import Instruction
from flagquantum.ecosystem import InteropConversionError
from flagquantum.ecosystem._qmod.source import load_qmod
from flagquantum.simulation.mps.entrypoints import run_mps
from flagquantum.simulation.mps.state import MPSState
from tools._qmod_basis_mps import probe_basis_mps

pytestmark = pytest.mark.integration
_DTYPE = torch.complex128
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"


def _original() -> Path:
    manifest = json.loads((_FIXTURES / "wide_cnf_sources.json").read_text())
    [row] = manifest["fixtures"]
    path = _FIXTURES / row["file"]
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == row["git_blob_sha1"]
    )
    license_bytes = (_FIXTURES / manifest["license_file"]).read_bytes()
    assert hashlib.sha256(license_bytes).hexdigest() == manifest["license_sha256"]
    return path


def _basis_and_phase(state: MPSState) -> tuple[tuple[int, ...], complex]:
    assert state.max_bond == 1
    assert not state.truncation_records
    bits, phase = [], 1.0 + 0j
    for site in state.tensors:
        zero, one = (complex(item) for item in site.reshape(2))
        assert (zero == 0) != (one == 0)
        bits.append(int(zero == 0))
        phase *= one if zero == 0 else zero
    return tuple(bits), phase


def test_original_preserves_all_clauses_with_exact_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _original()
    clauses = tuple(
        tuple(map(int, clause))
        for clause in re.findall(
            r"\(\(qbv\[(\d+)\] or \(not qbv\[(\d+)\]\)\) or qbv\[(\d+)\]\)",
            path.read_text(),
        )
    )
    assert len(clauses) == 67
    assert (
        {a for a, _, _ in clauses}
        == {b for _, b, _ in clauses}
        == {c for _, _, c in clauses}
        == set(range(67))
    )
    imported = load_qmod(path)
    selection = select_register_program(
        imported.program, imported.functions, include_boolean=True
    )
    row = selection.selected
    assert row and row.target and row.arithmetic
    circuit = row.target.submitted
    wires = tuple(reversed(selection.registers["qbv"]))
    assert len(wires) == 67
    scratch = set(range(circuit.n_wires)) - set(wires)
    assert len(scratch) == row.arithmetic.peak_workspace == 83

    def forbid_dense(*args: object, **kwargs: object) -> NoReturn:
        pytest.fail("wide CNF materialized a dense statevector")

    monkeypatch.setattr(MPSState, "to_statevector", forbid_dense)
    evidence = probe_basis_mps(imported, 4800)
    assert evidence["stage"] == "execution_smoke_passed"
    assert evidence["basis_sign"] == -1
    assert evidence["basis_state"] == "0" * 150
    assert probe_basis_mps(imported, 4799)["stage"] == "planning_refused"

    for enabled in (set(), {clauses[0][1]}, set(range(67))):
        marked = all(
            a in enabled or b not in enabled or c in enabled for a, b, c in clauses
        )
        reference = fq.Circuit(circuit.n_wires, dtype=_DTYPE)
        prefix = tuple(Instruction("x", (wires[bit],)) for bit in sorted(enabled))
        for bit in sorted(enabled):
            reference.x(wires[bit])
        if marked:
            spare = min(scratch)
            reference.x(spare).z(spare).x(spare)
        actual = run_mps(
            replace(circuit, instructions=(*prefix, *circuit.instructions)),
            dtype=_DTYPE,
        )
        expected = run_mps(reference)
        bits, phase = _basis_and_phase(actual)
        assert (bits, phase) == _basis_and_phase(expected)
        assert all(bits[wire] == 0 for wire in scratch)


def _program(predicate: str) -> str:
    return f"""
        qfunc main(output q:qbit[4]) {{
            allocate(q); hadamard_transform(q);
            m:qnum; allocate(1,m);
            within {{ X(m); H(m); }} apply {{ m ^= {predicate}; }}
            free(m);
        }}
    """


def test_cnf_full_coherent_sign_with_tautology_and_duplicate() -> None:
    expression = (
        "(q[0] or not q[1] or q[0]) and (q[2] or q[1] or q[3]) and (q[3] or not q[3])"
    )
    program = synth.synthesize(_program(expression), source_format="qmod")
    reference = fq.Circuit(4, dtype=_DTYPE)
    for wire in range(4):
        reference.h(wire)
    signs = []
    for word in range(16):
        a, b, c, d = (bool(word & (1 << index)) for index in range(4))
        signs.append(-1 if (a or not b) and (c or b or d) else 1)
    reference.any(*range(4), matrix=torch.diag(torch.tensor(signs, dtype=_DTYPE)))
    actual = fq.run(program).statevector().reshape(16, -1)
    expected = fq.run(reference).statevector().reshape(-1)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    assert actual[:, 1:].count_nonzero().item() == 0
    with pytest.raises(ValueError, match="workspace|budget"):
        synth.synthesize(_program(expression), source_format="qmod", max_workspace=0)


@pytest.mark.parametrize(
    ("predicate", "message"),
    [
        ("q[4] and q[0]", "index is out of bounds"),
        ("q[0] and m", "requires disjoint indexed bits"),
        ("q[0] and (q[0] or q[1] or q[2] or q[3])", "at most three literals"),
        (" and ".join(["q[0]"] * 81), "eighty clauses"),
    ],
)
def test_cnf_rejects_unproved_or_over_budget_predicates(
    predicate: str, message: str
) -> None:
    with pytest.raises(InteropConversionError, match=message):
        synth.synthesize(_program(predicate), source_format="qmod")
