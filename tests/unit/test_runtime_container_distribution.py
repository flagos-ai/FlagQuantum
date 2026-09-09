from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_runtime_image_installs_only_the_prebuilt_package() -> None:
    dockerfile = (ROOT / "docker/runtime/Dockerfile").read_text(encoding="utf-8")

    assert "cudnn9-runtime" in dockerfile
    assert "pip install --no-deps" in dockerfile
    assert ".[dev" not in dockerfile
    assert "jax" not in dockerfile.lower()
    assert "build-essential" not in dockerfile


def test_jiuding_recipe_obeys_platform_build_contract() -> None:
    dockerfile = (ROOT / "docker/runtime/Dockerfile.jiuding").read_text(
        encoding="utf-8"
    )

    assert dockerfile.startswith("FROM pytorch/pytorch:")
    assert dockerfile.count("COPY ") == 1
    assert "openssh-server" in dockerfile
    assert "ENTRYPOINT" not in dockerfile
    assert "torch.version.cuda" in dockerfile
    assert 'org.flagquantum.sc27.release_evidence="false"' in dockerfile
