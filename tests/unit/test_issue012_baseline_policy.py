from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


SMOKE_PATH = Path("tests/smoke/test_smoke_baseline.py")
UNIT_PATH = Path("tests/unit/test_unit_baseline.py")
BASELINE_PATHS = (SMOKE_PATH, UNIT_PATH)


def test_issue012_baseline_files_are_marked_for_default_selection():
    smoke_text = SMOKE_PATH.read_text(encoding="utf-8")
    unit_text = UNIT_PATH.read_text(encoding="utf-8")

    assert "pytestmark = pytest.mark.smoke" in smoke_text
    assert "pytestmark = pytest.mark.unit" in unit_text


def test_issue012_smoke_baseline_covers_public_api_expectation_and_autograd():
    text = SMOKE_PATH.read_text(encoding="utf-8")

    assert "import flagquantum as fq" in text
    assert "fq.Circuit" in text
    assert "expectation_z" in text
    assert ".backward()" in text
    assert "requires_grad=True" in text


def test_issue012_unit_baseline_covers_pure_logic_planner_audit_and_runtime_metadata():
    text = UNIT_PATH.read_text(encoding="utf-8")

    assert "to_ir" in text
    assert "simple_compile" in text
    assert "plan_runtime_selection" in text
    assert "audit_distributed_scalability" in text
    assert "backend_execution_options" in text
    assert "single_device_fast_path" in text


def test_issue012_baseline_avoids_distributed_gpu_benchmark_and_subprocess_dependencies():
    forbidden_tokens = (
        "subprocess",
        "multiprocessing",
        "torchrun",
        "pytest.mark.gpu",
        "pytest.mark.distributed",
        "benchmarks.",
        "benchmark_results",
        "cuda",
    )

    for path in BASELINE_PATHS:
        text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in text, f"{path} should not depend on {token}"


def test_issue012_docs_and_agents_keep_smoke_unit_as_minimum_entry_point():
    testing = Path("docs/development/TESTING.md").read_text(encoding="utf-8")
    agents = Path("AGENTS.md").read_text(encoding="utf-8")
    assert 'python -m pytest -m "smoke or unit" -q' in testing
    assert 'python -m pytest -m "smoke or unit" -q' in agents
    normalized_testing = " ".join(testing.split())
    assert (
        "It must stay fast, local, and free of torchrun, GPU, benchmark generation, or distributed setup"
        in normalized_testing
    )
    assert "minimum verification entry point for every change" in normalized_testing
