from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
ADR_ROOT = ROOT / "docs" / "architecture" / "decisions"


def _read(name: str) -> str:
    return " ".join((ADR_ROOT / name).read_text(encoding="utf-8").split())


def test_phase1_contract_adrs_remain_proposals_with_required_sections() -> None:
    names = [
        "ARCH_002_ARTIFACT_METADATA_AUTHORITY.md",
        "ARCH_003_CAPABILITY_REQUIREMENT_DISCOVERY_EVIDENCE.md",
        "ARCH_004_EXECUTION_REQUEST_POLICY_BOUNDARY.md",
        "ARCH_005_EXECUTION_RESULT_EVIDENCE_COMPATIBILITY.md",
        "ARCH_007_SIMULATION_RUNTIME_PLANNING_BOUNDARY.md",
        "ARCH_008_MIGRATION_AND_VERIFICATION_PATHS.md",
    ]
    for name in names:
        text = _read(name)
        assert "Status: Proposed" in text
        for section in (
            "## Context",
            "## Decision Candidates",
            "## Prohibited Practices",
            "## Compatibility",
            "## Acceptance Tests",
            "## Open Questions",
        ):
            assert section in text, (name, section)


def test_superseded_provider_admission_model_points_to_current_decision() -> None:
    text = _read("ARCH_006_PROVIDER_ADMISSION_MODEL.md")
    assert "Status: Superseded by ARCH-001" in text
    assert "Compute and Remote" in text


def test_evidence_levels_and_fact_exposure_are_distinct_from_support_status() -> None:
    text = _read("ARCH_003_CAPABILITY_REQUIREMENT_DISCOVERY_EVIDENCE.md")
    for value in ("basic", "observable", "certification"):
        assert value in text
    for value in ("observed", "declared", "not_exposed", "unknown", "not_applicable"):
        assert value in text
    for value in ("unmeasured", "unsupported", "verified"):
        assert value in text
    assert "two orthogonal axes" in text
    assert "Capability or release claim levels must not exceed" in text
    assert "must not hide known or observed" in text


def test_program_artifact_v1_remains_the_only_envelope_authority() -> None:
    text = _read("ARCH_002_ARTIFACT_METADATA_AUTHORITY.md")
    assert "`ProgramArtifact` v1 remains the sole artifact envelope authority" in text
    assert "Any v2 must evolve within the same contract lineage" in text
    assert "Do not create a second parallel envelope" in text
    assert (
        "When this proposal was written, v1's only active production consumer chain"
        in text
    )
    for legacy in (
        "`SealedExecutableArtifact`",
        "`SealedCircuitIRRoundTrip`",
        "`DeploymentPackage`",
    ):
        assert legacy in text
    assert "cannot losslessly replace" in text


def test_program_artifact_v1_hash_and_metadata_facts_are_not_reinterpreted() -> None:
    text = _read("ARCH_002_ARTIFACT_METADATA_AUTHORITY.md")
    assert "recomputed from canonical JSON of the complete `to_dict()` output" in text
    assert "`metadata` all participate in the existing" in text
    assert "not a declared field transmitted with the envelope" in text
    assert "The receiver recomputes it from the complete received envelope" in text
    assert "arbitrary objects with callable `to_dict()`" in text
    assert "value domain is not closed" in text
    assert "This is not current v1 behavior" in text
    assert (
        "Do not describe v1 metadata as presentation-only or excluded from hashing"
        in text
    )


def test_migration_dispositions_and_verification_paths_are_bounded() -> None:
    text = _read("ARCH_008_MIGRATION_AND_VERIFICATION_PATHS.md")
    for value in ("migrate", "adapt", "freeze_legacy", "retire"):
        assert f"`{value}`" in text
    for field in (
        "ongoing maintenance cost",
        "one-time migration cost",
        "regression risk",
        "impact of retaining the implementation",
        "review date",
    ):
        assert field in text
    for path in ("Fast path", "Standard path", "Certification path"):
        assert path in text
    assert "An overdue review blocks" in text
    assert "Only changes to public semantics enter the Core" in text


def test_simulation_runtime_and_agent_boundaries_are_explicit() -> None:
    planning = _read("ARCH_007_SIMULATION_RUNTIME_PLANNING_BOUNDARY.md")
    for value in (
        "workload features",
        "algorithm constraints",
        "resource requirements",
        "candidate partitionings",
        "cost estimates",
    ):
        assert value in planning
    assert "Layers define ownership boundaries, not information isolation" in planning
    assert "Runtime combines these candidates with Platform" in planning

    providers = _read("ARCH_006_PROVIDER_ADMISSION_MODEL.md")
    assert "Agent-facing deterministic application services" in providers
    assert "LLM and" in providers and "external Compute Service" in providers
    assert "must not bypass" in providers


def test_adr_index_maps_rules_to_machine_and_ci_entry_points() -> None:
    text = _read("README.md")
    for value in (
        "team-ownership.toml",
        "architecture.toml",
        "capability-maturity.toml",
        "tools/check_team_scope.py",
        "tools/check_architecture.py",
        "tools/docs_source_of_truth.py --check",
        "tools/ci_tier.py",
    ):
        assert value in text
    assert "Unimplemented checks are not described as enabled" in text
