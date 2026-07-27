from pathlib import Path

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


def test_issue011_registers_tiered_pytest_markers():
    text = Path("pytest.ini").read_text(encoding="utf-8")

    assert "--strict-markers" in text
    for marker in TIERED_MARKERS:
        assert f"    {marker}:" in text


def test_issue011_testing_policy_documents_marker_boundaries():
    text = Path("docs/TESTING.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    for marker in TIERED_MARKERS:
        assert f"`{marker}`" in text
    assert "does not prove real multi-GPU or multi-node capacity expansion" in text
    assert "must not be used as release-grade scalability evidence" in text
    assert "Do not treat an empty marker selection as verification" in text
    assert "Current Runnable Commands" in text
    assert 'python -m pytest -m "smoke or unit" -q' in text
    assert "Planned Marker Commands" in text
    assert 'python -m pytest -m "distributed_cpu" -q' in text
    assert 'python -m pytest -m "benchmark_contract or release_gate" -q' in text
    assert "benchmark generation remains an explicit command" in normalized


def test_issue011_agents_guidance_uses_tiered_strategy_without_false_scalability_claims():
    text = Path("AGENTS.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "Use the tiered pytest policy in `docs/TESTING.md`" in text
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
