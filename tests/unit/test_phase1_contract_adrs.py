from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
ADR_ROOT = ROOT / "docs" / "architecture" / "decisions"


def _read(name: str) -> str:
    return (ADR_ROOT / name).read_text(encoding="utf-8")


def test_phase1_contract_adrs_remain_proposals_with_required_sections():
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
        assert "状态：Proposed" in text
        for section in (
            "## 上下文",
            "## 决策候选",
            "## 禁止事项",
            "## 兼容性",
            "## 验收测试",
            "## 未决问题",
        ):
            assert section in text, (name, section)


def test_superseded_provider_admission_model_points_to_current_decision():
    text = _read("ARCH_006_PROVIDER_ADMISSION_MODEL.md")
    assert "状态：Superseded by ARCH-001" in text
    assert "Compute 与 Remote" in text


def test_evidence_levels_and_fact_exposure_are_distinct_from_support_status():
    text = _read("ARCH_003_CAPABILITY_REQUIREMENT_DISCOVERY_EVIDENCE.md")
    for value in ("basic", "observable", "certification"):
        assert value in text
    for value in ("observed", "declared", "not_exposed", "unknown", "not_applicable"):
        assert value in text
    for value in ("unmeasured", "unsupported", "verified"):
        assert value in text
    assert "两条正交轴" in text
    assert "能力或发布声明的等级不得高于" in text
    assert "不得隐瞒已经知道或观察到的" in text


def test_program_artifact_v1_remains_the_only_envelope_authority():
    text = _read("ARCH_002_ARTIFACT_METADATA_AUTHORITY.md")
    assert "`ProgramArtifact` v1 继续作为唯一 artifact envelope 权威" in text
    assert "任何 v2 都是同一契约" in text
    assert "不得并行创建第二套" in text
    assert "本提案形成时，v1 唯一实际运行的生产消费链" in text
    for legacy in (
        "`SealedExecutableArtifact`",
        "`SealedCircuitIRRoundTrip`",
        "`DeploymentPackage`",
    ):
        assert legacy in text
    assert "不能无损替代" in text


def test_program_artifact_v1_hash_and_metadata_facts_are_not_reinterpreted():
    text = _read("ARCH_002_ARTIFACT_METADATA_AUTHORITY.md")
    assert "完整 `to_dict()` canonical JSON 重算" in text
    assert "`metadata` 均参与现有" in text
    assert "不作为声明字段随 envelope 传输" in text
    assert "接收方从接收到的完整" in text
    assert "任意对象的 callable `to_dict()`" in text
    assert "值域未闭合" in text
    assert "这不是 v1 当前行为" in text
    assert "不将 v1 metadata 描述为“不参与 hash”" in text


def test_migration_dispositions_and_verification_paths_are_bounded():
    text = _read("ARCH_008_MIGRATION_AND_VERIFICATION_PATHS.md")
    for value in ("migrate", "adapt", "freeze_legacy", "retire"):
        assert f"`{value}`" in text
    for field in (
        "持续维护成本",
        "一次性迁移成本",
        "回归风险",
        "继续存续的影响",
        "复核时间",
    ):
        assert field in text
    for path in ("快速路径", "标准路径", "认证路径"):
        assert path in text
    assert "逾期未复核即阻塞" in text
    assert "只有公共语义变化才进入 Core" in text


def test_simulation_runtime_and_agent_boundaries_are_explicit():
    planning = _read("ARCH_007_SIMULATION_RUNTIME_PLANNING_BOUNDARY.md")
    for value in (
        "workload features",
        "algorithm constraints",
        "resource requirements",
        "candidate partitionings",
        "cost estimates",
    ):
        assert value in planning
    assert "分层是所有权边界，不是信息隔离" in planning
    assert "Runtime 将这些候选与 Platform" in planning

    providers = _read("ARCH_006_PROVIDER_ADMISSION_MODEL.md")
    assert "Agent-facing deterministic application services" in providers
    assert "LLM 与" in providers and "外部 Compute Service" in providers
    assert "不得绕过" in providers


def test_adr_index_maps_rules_to_machine_and_ci_entry_points():
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
    assert "不把尚未实现的检查写成已启用" in text
