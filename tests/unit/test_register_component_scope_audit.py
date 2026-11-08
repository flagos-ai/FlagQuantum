"""Reproduce missing register semantics before extending the shared boundary.

These are audit witnesses, not acceptance of a native register graph. The source
witnesses establish requirements for the forthcoming contract conformance suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.audit_register_component_scope import audit_program_scope

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_audit_preserves_the_existing_static_and_dynamic_boundaries():
    protected = ROOT / "contracts/multi-level-ir-internal-v1-candidate.json"
    original = protected.read_bytes()
    report = audit_program_scope(ROOT)
    assert protected.read_bytes() == original
    assert json.loads(json.dumps(report)) == report
    for result in report["round_trips"].values():
        assert result == {
            "lowering_state": "supported_exact",
            "restoration_state": "supported_exact",
            "diagnostics": [],
            "verified": True,
            "canonical_payload_equal": True,
        }


def test_saved_evidence_describes_the_current_scope():
    evidence = ROOT / "docs/development/evidence/register_component_scope_audit.json"
    assert audit_program_scope(ROOT) == json.loads(evidence.read_text())


@pytest.mark.ftqc_reference
def test_source_zero_width_is_a_logical_port_not_always_an_absent_register():
    from qualtran import QAny, Register, Signature

    explicit = Register("empty", QAny(0))
    signature = Signature((explicit,))
    assert tuple(signature.lefts()) == tuple(signature.rights()) == (explicit,)
    assert list(explicit.all_idxs()) == [()]
    assert explicit.total_qubits() == 0
    assert list(Signature.build(empty=0)) == []
    shaped = Register("empty", QAny(3), shape=(0, 2))
    assert list(shaped.all_idxs()) == []
    assert shaped.total_qubits() == 0


@pytest.mark.ftqc_reference
def test_source_same_name_on_opposite_sides_does_not_mean_one_register():
    from qualtran import QAny, Register, Side, Signature

    before = Register("word", QAny(2), side=Side.LEFT)
    after = Register("word", QAny(3), side=Side.RIGHT)
    signature = Signature((before, after))
    assert signature.get_left("word") is before
    assert signature.get_right("word") is after
    assert list(signature.groups()) == [("word", [before, after])]
    assert signature.n_qubits() == 3
    assert signature.adjoint().get_left("word").bitsize == 3
    with pytest.raises(ValueError, match="more than once per side"):
        Signature((before, before))


@pytest.mark.ftqc_reference
def test_source_shape_order_and_symbolic_rejection_are_distinct_from_width():
    from qualtran import QAny, Register
    from sympy import Symbol

    register = Register("words", QAny(8), shape=(2, 3))
    assert list(register.all_idxs()) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert register.total_qubits() == 48
    n = Symbol("n", integer=True, positive=True)
    symbolic = Register("words", QAny(8), shape=(n, 3))
    assert symbolic.shape_symbolic == (n, 3)
    assert symbolic.total_qubits() == 24 * n
    with pytest.raises(ValueError, match="Cannot get real-valued shape"):
        _ = symbolic.shape
    # A symbolic element width does not make a concrete shape unindexable.
    width_only = Register("words", QAny(n), shape=(2, 3))
    assert width_only.shape == (2, 3)
    assert list(width_only.all_idxs()) == list(register.all_idxs())
