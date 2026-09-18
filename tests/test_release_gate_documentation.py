from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

TIERED_MARKERS = {
    "smoke",
    "unit",
    "integration",
    "distributed_cpu",
    "distributed_accel",
    "distributed_multinode",
    "benchmark_contract",
    "release_gate",
    "scalability",
}


def test_registers_tiered_pytest_markers():
    text = Path("pytest.ini").read_text(encoding="utf-8")

    assert "--strict-markers" in text
    for marker in TIERED_MARKERS:
        assert f"    {marker}:" in text


def test_testing_policy_documents_marker_boundaries():
    text = Path("docs/development/TESTING.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    for marker in TIERED_MARKERS:
        assert f"`{marker}`" in text
    assert "does not prove real multi-GPU or multi-node capacity expansion" in text
    assert "must not be used as release-grade scalability evidence" in text
    assert "Do not treat an empty marker selection as verification" in text
    # The manual used to carry a `## Current Runnable Commands` heading with
    # nothing under it, immediately followed by `## Quick Start`. This assertion
    # pinned the empty heading; it now pins the section that actually lists the
    # commands a reader can run.
    assert "## Quick Start" in text
    assert 'python -m pytest -m "smoke or unit" -q' in text
    assert "Planned Marker Commands" in text
    assert 'python -m pytest -m "distributed_cpu" -q' in text
    assert 'python -m pytest -m "benchmark_contract or release_gate" -q' in text
    assert "benchmark generation remains an explicit command" in normalized


def test_agents_guidance_uses_tiered_strategy_without_false_scalability_claims():
    text = Path("AGENTS.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "Use the tiered pytest policy in `docs/development/TESTING.md`" in text
    assert 'python -m pytest -m "smoke or unit" -q' in text
    assert "Only run marker-selection commands that have seeded tests" in text
    assert (
        "CPU distributed tests prove semantics and fail-closed behavior only"
        in normalized
    )
    assert (
        "Never use CPU distributed tests alone as scalability release evidence"
        in normalized
    )
    assert "fixed file lists become examples under a tiered strategy" not in normalized
