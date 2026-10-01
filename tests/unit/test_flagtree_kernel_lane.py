"""Static contracts for the real-device FlagTree compatibility lane."""

from pathlib import Path

import pytest

from flagquantum.kernels.catalog import EVIDENCE, IMPLEMENTATIONS

pytestmark = pytest.mark.unit

ROOT = Path(__file__).parents[2]
LANE = ROOT / "tests" / "gpu" / "flagtree"


def test_flagtree_suite_covers_every_triton_evidence_file() -> None:
    triton_implementation_ids = {
        implementation.implementation_id
        for implementation in IMPLEMENTATIONS
        if implementation.provider == "triton"
    }
    required_files = {
        node_id.partition("::")[0]
        for evidence in EVIDENCE
        if evidence.implementation_id in triton_implementation_ids
        for node_id in (
            evidence.correctness_tests
            + evidence.gradient_tests
            + evidence.capability_tests
        )
    }
    suite_files = {
        line
        for line in (LANE / "suite.txt").read_text(encoding="utf-8").splitlines()
        if line.startswith("tests/")
    }

    assert "tests/gpu/test_flagtree_kernel_provider.py" in suite_files
    assert required_files <= suite_files


def test_flagtree_image_is_pinned_and_replaces_stock_triton() -> None:
    dockerfile = (LANE / "Dockerfile").read_text(encoding="utf-8")

    assert "vllm-openai@sha256:" in dockerfile
    assert "ARG FLAGTREE_VERSION=0.7.0" in dockerfile
    assert dockerfile.index("pip uninstall -y triton") < dockerfile.index(
        '"flagtree==${FLAGTREE_VERSION}"'
    )
    assert 'packages_distributions().get("triton") == ["flagtree"]' in dockerfile
