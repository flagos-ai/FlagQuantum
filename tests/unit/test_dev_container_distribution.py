from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_development_image_is_fail_closed_for_sc27_evidence() -> None:
    dockerfile = (ROOT / "docker/dev/Dockerfile").read_text(encoding="utf-8")

    assert 'org.flagquantum.image.class="development_only"' in dockerfile
    assert 'org.flagquantum.sc27.release_evidence="false"' in dockerfile
    assert "org.opencontainers.image.source" in dockerfile


def test_ghcr_workflow_publishes_private_multiarch_cpu_tags() -> None:
    workflow = (ROOT / ".github/workflows/publish-dev-container.yml").read_text(
        encoding="utf-8"
    )

    assert "packages: write" in workflow
    assert "linux/amd64,linux/arm64" in workflow
    assert "type=raw,value=cpu" in workflow
    assert "type=sha,prefix=cpu-sha-" in workflow
    assert 'test "$visibility" = private' in workflow
    assert "provenance: mode=max" in workflow
    assert "sbom: true" in workflow


def test_remote_publisher_keeps_cuda_and_cpu_namespaces_separate() -> None:
    publisher = (ROOT / "tools/containers/publish_dev_image.sh").read_text(
        encoding="utf-8"
    )

    assert 'current_tag="$image:cuda-amd64"' in publisher
    assert 'version_tag="$image:cuda-$version"' in publisher
    assert '--tag "$image:cpu"' in publisher
    assert '--tag "$image:cpu-$version"' in publisher
