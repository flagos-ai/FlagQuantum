from pathlib import Path

import pytest

from tools.ci_tier import CI_TIERS

pytestmark = pytest.mark.unit


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_testing_manual_has_human_entrypoints():
    text = _read("docs/development/TESTING.md")
    for heading in (
        "## Quick Start",
        "## Marker Reference",
        "## Test Tiers",
        "## CI Policy",
        "## Path Classification Mapping",
        "## Non-Negotiable Release Boundary",
    ):
        assert heading in text


def test_testing_manual_distinguishes_required_workflows():
    text = _read("docs/development/TESTING.md")
    for phrase in (
        "Daily Development",
        "Local API And Runtime Work",
        "Distributed CPU Semantics",
        "Real Multi-GPU Or Accelerator Work",
        "Multi-Node Work",
        "Benchmark Release Validation",
    ):
        assert phrase in text


def test_testing_manual_lists_current_commands():
    text = _read("docs/development/TESTING.md")
    for command in (
        "python tools/ci_tier.py pr-default",
        "python tools/ci_tier.py pr-runtime",
        "python tools/ci_tier.py pr-distributed",
        "python tools/ci_tier.py gpu-scheduled",
        "python tools/ci_tier.py multinode-scheduled",
        "python tools/ci_tier.py release",
        'python -m pytest -m "smoke or unit" -q',
        'python -m pytest -m "integration" -q',
        'python -m pytest -m "distributed_cpu or release_gate or benchmark_contract" -q',
        'python -m pytest -m "distributed_accel and gpu" -q',
        'python -m pytest -m "distributed_multinode" -q',
        "python benchmarks/audit_results.py --input benchmarks/results",
        "python benchmarks/audit_results.py --input benchmarks/results/scalability --require-scalability",
    ):
        assert command in text


def test_testing_manual_documents_every_marker_with_proof_boundary():
    text = _read("docs/development/TESTING.md")
    for marker in (
        "smoke",
        "unit",
        "integration",
        "distributed_cpu",
        "distributed_accel",
        "distributed_multinode",
        "benchmark_contract",
        "release_gate",
        "scalability",
        "gpu",
        "distributed",
        "slow",
    ):
        assert f"| `{marker}` |" in text
    assert "| Marker | Runtime Environment | Proves | Does Not Prove |" in text
    assert "Real benchmark performance or hardware behavior" in text
    assert "Release readiness or scalability on its own" in text


def test_testing_manual_matches_ci_tier_commands_and_boundaries():
    text = _read("docs/development/TESTING.md")
    normalized = " ".join(text.split())

    for tier_name, tier in CI_TIERS.items():
        assert f"python tools/ci_tier.py {tier_name}" in text
        for command_line in tier.command_lines():
            if tier_name in {"nightly", "release"} and command_line.startswith(
                "python -m pytest"
            ):
                continue
            if command_line.startswith("python -m pytest"):
                quoted_marker = command_line.replace(
                    "python -m pytest -m ", 'python -m pytest -m "'
                ).replace(" -q", '" -q')
                assert command_line in normalized or quoted_marker in text, (
                    tier_name,
                    command_line,
                )
            else:
                assert command_line in text, (tier_name, command_line)
        assert (
            tier.does_not_prove.split(".")[0] in normalized
            or "not scalability evidence" in normalized
        )


def test_testing_manual_preserves_scalability_boundary():
    text = " ".join(_read("docs/development/TESTING.md").split())
    assert "CPU distributed tests prove semantics and fail-closed behavior only" in text
    assert "They do not prove real multi-card capacity expansion" in text
    assert (
        "Never use CPU distributed tests alone as scalability release evidence" in text
    )
    assert (
        "Release-grade scalability requires one logical workload sharded across ranks"
        in text
    )
    assert 'distribution_semantics="sharded_across_ranks"' in text
    assert "sharded gradient/optimizer ownership" in text


def test_testing_manual_aligned_with_agents():
    docs = _read("docs/development/TESTING.md")
    agents = _read("AGENTS.md")
    normalized_docs = " ".join(docs.split())
    assert "python tools/ci_tier.py pr-default" in agents
    assert "python tools/ci_tier.py pr-default" in docs
    assert (
        "Never use CPU distributed tests alone as scalability release evidence"
        in agents
    )
    assert (
        "Never use CPU distributed tests alone as scalability release evidence"
        in normalized_docs
    )
