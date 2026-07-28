from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_issue013_registers_distributed_marker_split():
    text = _read("pytest.ini")

    assert "    distributed:" in text
    assert "    distributed_cpu:" in text
    assert "    distributed_accel:" in text
    assert "    distributed_multinode:" in text
    assert "    gpu:" in text


def test_issue013_cpu_distributed_layer_contains_statevector_fail_closed_coverage():
    cpu_files = [
        "tests/test_distributed_backend_policy.py",
        "tests/test_distributed_scalability_audit.py",
        "tests/test_distributed_statevector.py",
        "tests/test_jax_distributed_plan.py",
    ]

    for path in cpu_files:
        text = _read(path)
        assert "pytest.mark.distributed" in text, path
        assert "pytest.mark.distributed_cpu" in text, path

    audit_text = _read("tests/test_distributed_scalability_audit.py")
    assert "scalability_claim_allowed" in audit_text
    assert "require_distributed_scalability" in audit_text
    assert "replicated" in audit_text


def test_issue013_accelerator_tests_are_not_selected_by_cpu_marker():
    accelerator_files = [
        "tests/distributed/test_quantum_device.py",
        "tests/distributed/test_quantum_gradients.py",
    ]

    for path in accelerator_files:
        text = _read(path)
        assert "pytest.mark.distributed" in text, path
        assert "pytest.mark.distributed_accel" in text, path
        assert "pytest.mark.gpu" in text, path
        assert "pytest.mark.distributed_cpu" not in text, path
        assert '"RANK" not in os.environ' in text, path


def test_issue013_torchrun_cpu_candidates_keep_local_skip_guards():
    torchrun_cpu_files = [
        "tests/distributed/test_hybrid_jax_runtime.py",
        "tests/distributed/test_runtime_modes.py",
        "tests/distributed/test_statevector_correctness.py",
    ]

    for path in torchrun_cpu_files:
        text = _read(path)
        assert "pytest.mark.distributed" in text, path
        assert "pytest.mark.distributed_cpu" in text, path
        assert "pytest.mark.distributed_multinode" in text, path
        assert '"RANK" not in os.environ' in text, path
        assert (
            "requires torchrun with distributed RANK/WORLD_SIZE environment" in text
        ), path


def test_issue013_docs_and_agents_reject_cpu_distributed_release_evidence():
    docs = _read("docs/development/TESTING.md")
    agents = _read("AGENTS.md")
    normalized_agents = " ".join(agents.split())

    assert 'python -m pytest -m "distributed_cpu" -q' in docs
    assert 'python -m pytest -m "distributed_cpu" -q' in agents
    assert (
        "CPU distributed tests prove semantics and fail-closed behavior only"
        in normalized_agents
    )
    assert "must never be cited as scalability release evidence" in normalized_agents
    normalized_docs = " ".join(docs.split())
    assert "They do not prove real multi-card capacity expansion" in normalized_docs
    assert "release-grade scalability evidence" in normalized_docs
