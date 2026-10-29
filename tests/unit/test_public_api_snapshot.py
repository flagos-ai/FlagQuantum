from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import public_api_snapshot

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_public_api_matches_reviewed_retained_baseline() -> None:
    assert public_api_snapshot.validate() == ()


def test_release_version_literal_is_not_pinned_to_v02(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = public_api_snapshot.generate()
    changed["exports"]["__version__"] = {
        "kind": "value",
        "qualified_type": "builtins.str",
        "value": {"kind": "literal", "value": "99.0.0rc1"},
    }
    monkeypatch.setattr(public_api_snapshot, "generate", lambda: changed)

    assert public_api_snapshot.validate() == ()


def test_baseline_covers_current_stable_export_manifest() -> None:
    manifest = json.loads((ROOT / "docs/public_api_v1.json").read_text())
    baseline = json.loads(
        (ROOT / "contracts/public-api-v0.2-baseline.json").read_text()
    )
    options = json.loads(
        (ROOT / "contracts/execution-options-v1-candidate.json").read_text()
    )
    plan = json.loads((ROOT / "contracts/execution-plan-v1-candidate.json").read_text())
    errors_module = json.loads(
        (ROOT / "contracts/errors-module-boundary-v1-candidate.json").read_text()
    )
    extensions = json.loads(
        (ROOT / "contracts/extension-protocol-v1-candidate.json").read_text()
    )
    observable_outputs = json.loads(
        (ROOT / "contracts/observable-outputs-v1-candidate.json").read_text()
    )
    twin = json.loads((ROOT / "contracts/twin-v1-candidate.json").read_text())
    authorized_additions = (
        {options["root_addition"]}
        if options["root_manifest_authorized"] is True
        else set()
    )
    if plan["root_manifest_authorized"] is True:
        authorized_additions.add(plan["root_addition"])
    if errors_module["implementation_authorized"] is True:
        authorized_additions.update(errors_module["stable_extension"]["additions"])
    if (
        extensions["implementation_authorized"] is True
        and extensions["root_manifest_change"] is True
    ):
        authorized_additions.update(extensions["root_additions"])
    authorized_additions.update(observable_outputs["root_additions"])
    jobs_contract = json.loads(
        (ROOT / "contracts/remote-jobs-v1-candidate.json").read_text()
    )
    if jobs_contract["implementation_authorized"] is True:
        authorized_additions.update(jobs_contract["root_additions"])
    if (
        twin["implementation_authorized"] is True
        and twin["root_manifest_authorized"] is True
    ):
        authorized_additions.add(twin["root_addition"])
    gradient = json.loads(
        (ROOT / "contracts/gradient-api-v1-candidate.json").read_text()
    )
    if gradient["implementation_authorized"] is True:
        authorized_additions.update(gradient["root_additions"])
    openqasm_import = json.loads(
        (ROOT / "contracts/openqasm-import-v1-candidate.json").read_text()
    )
    if openqasm_import["implementation_authorized"] is True:
        authorized_additions.update(openqasm_import["root_additions"])

    experiment = json.loads(
        (ROOT / "contracts/experiment-file-v1-candidate.json").read_text()
    )
    if experiment["implementation_authorized"] is True:
        authorized_additions.update(experiment["root_additions"])

    assert (
        set(manifest["stable_exports"])
        <= set(baseline["exports"]) | authorized_additions
    )
    assert baseline["status"] == "pre_open_source_migration_baseline"


def test_snapshot_diagnostic_forbids_unreviewed_regeneration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = public_api_snapshot.generate()
    changed["exports"]["Circuit"] = {"changed": True}
    monkeypatch.setattr(public_api_snapshot, "generate", lambda: changed)

    errors = public_api_snapshot.validate()

    assert len(errors) == 1
    assert "do not regenerate" in errors[0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("operation", "ising.sample_pair"),
        ("production_implemented", False),
        ("braid_only_source_and_plan_version", "1.2"),
        ("new_operation_source_and_plan_version", "1.1"),
        ("unapproved_field", "anything"),
    ],
)
def test_unread_contract_authorization_does_not_accept_unreviewed_changes(
    field: str, value: object
) -> None:
    contract = json.loads(
        (ROOT / "contracts/execution-plan-v1-candidate.json").read_text()
    )
    contract["native_ising_admission"]["unread_observation"][field] = value
    errors = public_api_snapshot._validate_authorized_execution_plan(
        contract, ["ExecutionPlan"]
    )
    assert errors == ["native Ising admission differs from approved proposals 085/089"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("operation", "ising.unapproved_sample"),
        ("production_implemented", False),
        ("environment_protocol_version", "1.0"),
        ("new_operation_source_and_plan_version", "1.2"),
        ("unapproved_field", "anything"),
    ],
)
def test_sampled_contract_authorization_rejects_unreviewed_changes(
    field: str, value: object
) -> None:
    contract = json.loads(
        (ROOT / "contracts/execution-plan-v1-candidate.json").read_text()
    )
    contract["native_ising_admission"]["sampled_observation"][field] = value
    assert public_api_snapshot._validate_authorized_execution_plan(
        contract, ["ExecutionPlan"]
    ) == ["native Ising admission differs from approved proposals 085/089"]


def test_sampled_source_signature_is_protected(monkeypatch: pytest.MonkeyPatch) -> None:
    from flagquantum.core.ising import IsingBraidProgram

    def unreviewed(self: IsingBraidProgram, *, outcome: str) -> IsingBraidProgram:
        raise AssertionError("signature inspection must not call this method")

    monkeypatch.setattr(IsingBraidProgram, "measure_pair", unreviewed)
    contract = json.loads(
        (ROOT / "contracts/execution-plan-v1-candidate.json").read_text()
    )
    assert public_api_snapshot._validate_authorized_execution_plan(
        contract, ["ExecutionPlan"]
    ) == [
        "native Ising sampled source must expose measure_pair(self) -> IsingBraidProgram"
    ]
