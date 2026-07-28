from pathlib import Path

import pytest

from tools.ci_tier import CI_TIERS

pytestmark = pytest.mark.unit


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_issue017_agents_uses_tiered_ci_entrypoints():
    text = _read("AGENTS.md")
    assert "python tools/ci_tier.py pr-default" in text
    assert "python tools/ci_tier.py pr-runtime" in text
    assert "python tools/ci_tier.py pr-distributed" in text
    assert "python tools/ci_tier.py gpu-scheduled" in text
    assert "python tools/ci_tier.py multinode-scheduled" in text
    assert "python tools/ci_tier.py release" in text


def test_issue017_agents_starts_small_then_expands_by_blast_radius():
    text = _read("AGENTS.md")
    normalized = " ".join(text.split())

    assert "run the smallest meaningful layer first" in normalized
    assert "expand by blast radius" in normalized
    assert text.index("python tools/ci_tier.py pr-default") < text.index(
        "python tools/ci_tier.py pr-runtime"
    )
    assert text.index("python tools/ci_tier.py pr-runtime") < text.index(
        "python tools/ci_tier.py pr-distributed"
    )
    assert CI_TIERS["pr-default"].blocks_default_pr is True
    assert CI_TIERS["gpu-scheduled"].blocks_default_pr is False
    assert CI_TIERS["multinode-scheduled"].blocks_default_pr is False


def test_issue017_agents_maps_path_classifications_to_tiers():
    text = _read("AGENTS.md")
    for phrase in (
        "`single_device_fast_path`",
        "`sharded_across_ranks`",
        "`rank_local_replicated_kernel`",
        "`replicated_per_rank`",
        "`manual_sliced_tensor_contraction`",
        "`observable_term_parallel`",
    ):
        assert phrase in text
    assert "`smoke`, `unit`, and local `integration`" in text
    assert "`distributed_cpu` plus `release_gate`" in text
    assert "`distributed_accel` and `gpu`" in text
    assert "`distributed_multinode`" in text


def test_issue017_agents_keeps_focused_legacy_examples():
    text = _read("AGENTS.md")
    assert "tests/test_native_circuit.py tests/test_backends.py" in text
    assert (
        "tests/test_distributed_statevector.py tests/test_jax_distributed_plan.py "
        "tests/test_distributed_scalability_audit.py"
    ) in text
    assert "tests/benchmark_contract" in text


def test_issue017_agents_rejects_cpu_distributed_as_release_evidence():
    text = " ".join(_read("AGENTS.md").split())
    assert "CPU distributed tests prove semantics" in text
    assert "must never be cited as scalability release evidence" in text
    assert (
        "Never use CPU distributed tests alone as scalability release evidence" in text
    )
    assert "Real multi-GPU or multi-node production claims require" in text
    assert "release payload validation" in text


def test_issue017_ci_tiers_match_agents_workflow_commands():
    agents = _read("AGENTS.md")
    expected = {
        "pr-default": "python -m pytest -m smoke or unit -q",
        "pr-runtime": "python -m pytest -m integration -q",
        "pr-distributed": "python -m pytest -m distributed_cpu -q",
        "gpu-scheduled": "python -m pytest -m distributed_accel and gpu -q",
        "multinode-scheduled": "python -m pytest -m distributed_multinode -q",
    }

    for tier, command in expected.items():
        assert f"python tools/ci_tier.py {tier}" in agents
        assert command in CI_TIERS[tier].command_lines()
    assert (
        "python benchmarks/audit_results.py --input benchmarks/results"
        in CI_TIERS["release"].command_lines()
    )
    assert (
        "python benchmarks/audit_results.py --input benchmarks/results/scalability --require-scalability"
        in CI_TIERS["release"].command_lines()
    )


def test_docs_record_workflow_update():
    docs = _read("docs/development/TESTING.md")
    normalized_docs = " ".join(docs.split())
    assert "## Agent Workflow" in docs
    assert "tier commands are the official vocabulary" in normalized_docs
    assert (
        "start from `python tools/ci_tier.py pr-default`, then expand by blast radius"
        in normalized_docs
    )
