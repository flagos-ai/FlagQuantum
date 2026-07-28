from pathlib import Path

import pytest

from tools.ci_tier import CI_TIERS

pytestmark = pytest.mark.unit


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_issue016_agents_recommended_local_tests_are_integration_marked():
    for path in ("tests/test_native_circuit.py", "tests/test_backends.py"):
        text = _read(path)
        assert "pytest.mark.integration" in text, path


def test_issue016_integration_files_do_not_claim_distributed_or_benchmark_tiers():
    forbidden_markers = (
        "pytest.mark.distributed",
        "pytest.mark.distributed_cpu",
        "pytest.mark.distributed_accel",
        "pytest.mark.distributed_multinode",
        "pytest.mark.benchmark_contract",
        "pytest.mark.release_gate",
        "pytest.mark.gpu",
        "pytest.mark.scalability",
    )

    for path in ("tests/test_native_circuit.py", "tests/test_backends.py"):
        text = _read(path)
        for marker in forbidden_markers:
            assert marker not in text, path


def test_issue016_agents_recommended_distributed_tests_are_cpu_marked():
    for path in (
        "tests/test_distributed_statevector.py",
        "tests/test_jax_distributed_plan.py",
        "tests/test_distributed_scalability_audit.py",
    ):
        text = _read(path)
        assert "pytest.mark.distributed" in text, path
        assert "pytest.mark.distributed_cpu" in text, path
        assert "pytest.mark.integration" not in text, path


def test_issue016_benchmark_contract_tests_are_release_gate_marked():
    text = _read("tests/benchmark_contract/test_benchmark_contract.py")
    assert "pytest.mark.benchmark_contract" in text
    assert "pytest.mark.release_gate" in text
    assert "pytest.mark.integration" not in text


def test_issue016_runtime_tier_uses_integration_marker_after_migration():
    runtime_tier = CI_TIERS["pr-runtime"]
    assert runtime_tier.command_lines() == ("python -m pytest -m integration -q",)
    assert "Local cross-module runtime/API behavior" in runtime_tier.proves
    assert "scalability claims" in runtime_tier.does_not_prove


def test_docs_record_incremental_marker_migration():
    docs = _read("docs/development/TESTING.md")
    assert 'python -m pytest -m "integration" -q' in docs
    assert "tests/test_native_circuit.py" in docs
    assert "tests/test_backends.py" in docs
