"""The OpenQASM import contract must fail closed on every claim it makes.

``tools/check_openqasm_import_contract.py`` reconciles
``contracts/openqasm-import-v1-candidate.json`` with the shipped importer. These
tests drive it with one deliberate drift per claim, so a gate that silently
stopped reading a field fails here rather than passing in CI.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tools.check_openqasm_import_contract import (
    RULE_READERS,
    construction_time_only,
    contract_errors,
    gate_tables_defined_once,
    ir_version_unchanged,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts" / "openqasm-import-v1-candidate.json"


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _errors(mutate: Callable[[dict[str, Any]], None]) -> tuple[str, ...]:
    contract = _contract()
    mutate(contract)
    return contract_errors(contract)


def test_the_checked_in_contract_matches_the_shipped_importer() -> None:
    assert contract_errors(_contract()) == ()


def test_the_declared_refusal_vocabulary_is_the_importer_s_vocabulary() -> None:
    """A code may not be declared without being raised, or raised undeclared."""

    assert any(
        "differs from the importer's" in error
        for error in _errors(lambda contract: contract["refusal_issue_codes"].pop())
    )
    assert any(
        "differs from the importer's" in error
        for error in _errors(
            lambda contract: contract["refusal_issue_codes"].append("invented_code")
        )
    )


def test_the_declared_refusal_vocabulary_stays_sorted_and_unique() -> None:
    assert any(
        "sorted order" in error
        for error in _errors(lambda contract: contract["refusal_issue_codes"].reverse())
    )
    assert any(
        "must not repeat" in error
        for error in _errors(
            lambda contract: contract["refusal_issue_codes"].append(
                contract["refusal_issue_codes"][0]
            )
        )
    )


def test_both_openqasm_version_lanes_are_required() -> None:
    """The contract cannot narrow the importer's lanes, or widen them."""

    assert any(
        "do not match the importer's" in error
        for error in _errors(lambda contract: contract["supported_versions"].pop())
    )
    assert any(
        "do not match the importer's" in error
        for error in _errors(
            lambda contract: contract["supported_versions"].append(3.1)
        )
    )


def test_a_rule_without_a_reader_is_rejected() -> None:
    """`rules` is prose only until something reads each key."""

    assert any(
        "rules with no reader" in error
        for error in _errors(
            lambda contract: contract["rules"].update({"fresh_rule": True})
        )
    )
    assert any(
        "names a rule the contract dropped" in error
        for error in _errors(
            lambda contract: contract["rules"].pop("ir_version_unchanged")
        )
    )


def test_every_rule_reader_names_something_that_exists() -> None:
    assert any(
        "which does not exist in" in error
        for error in contract_errors(_contract(), witness_tests={"test_unrelated"})
    )


def test_the_root_interchange_cannot_widen_or_narrow_quietly() -> None:
    assert any(
        "root_additions must be" in error
        for error in _errors(
            lambda contract: contract["root_additions"].append("emit_openqasm")
        )
    )
    assert any(
        "is not exported" in error
        for error in contract_errors(_contract(), exported=("plan", "run"))
    )


def test_the_declared_signature_is_the_exposed_signature() -> None:
    assert any(
        "signature does not match" in error
        for error in _errors(
            lambda contract: contract["signatures"].update(
                {"from_openqasm": "(source: bytes) -> str"}
            )
        )
    )


def test_an_unread_key_and_a_pinned_ir_version_are_both_rejected() -> None:
    """Every contract key has a reader, so an unknown key is a defect."""

    assert any(
        "unread keys" in error
        for error in _errors(lambda contract: contract.update({"ir_version": "1.0"}))
    )
    assert any(
        "unread keys" in error
        for error in _errors(lambda contract: contract.update({"extra": 1}))
    )


def test_the_approval_record_and_migration_paths_exist() -> None:
    assert any(
        "does not exist" in error
        for error in _errors(
            lambda contract: contract["approval"].update(
                {"approval_record": "docs/api-changes/absent.md"}
            )
        )
    )
    assert any(
        "does not exist" in error
        for error in _errors(
            lambda contract: contract.update(
                {"qubit_naming_migration": "docs/absent.md"}
            )
        )
    )


def test_authorization_is_required_before_the_gate_reads_the_interchange() -> None:
    assert any(
        "implementation_authorized" in error
        for error in _errors(
            lambda contract: contract.update({"implementation_authorized": False})
        )
    )


def test_an_undeclared_rule_reader_would_be_caught_by_this_gate() -> None:
    """`RULE_READERS` is the mapping the gate enforces, not a copy of it."""

    assert RULE_READERS, "the rule-reader mapping cannot be empty"
    assert all(readers for readers in RULE_READERS.values())


def test_the_importer_is_a_construction_time_reader() -> None:
    assert construction_time_only() == ()


def test_the_gate_tables_are_defined_exactly_once() -> None:
    assert gate_tables_defined_once() == ()


def test_the_interchange_added_no_ir_field() -> None:
    assert ir_version_unchanged(_contract()) == ()
