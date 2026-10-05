from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from examples.qdiffusion_kaiwu.build_environment_lock import build_environment_lock

pytestmark = pytest.mark.unit


def _wheel(root: Path, *, name: str, version: str, filename: str | None = None) -> Path:
    path = root / (filename or f"{name}-{version}-py3-none-any.whl")
    dist_info = f"{name.replace('-', '_')}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{dist_info}/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        )
    return path


def test_builder_binds_complete_runtime_to_reviewed_wheels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    numpy = _wheel(tmp_path, name="NumPy", version="2.2.6")
    torch = _wheel(tmp_path, name="torch", version="2.7.0")
    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.build_environment_lock._installed_distribution_inventory",
        lambda: {"numpy": ("2.2.6", "c" * 64), "torch": ("2.7.0", "d" * 64)},
    )
    output = tmp_path / "environment-lock.json"

    digest = build_environment_lock(
        artifacts=[torch.resolve(), numpy.resolve()], output=output.resolve()
    )

    record = json.loads(output.read_text(encoding="utf-8"))
    assert output.stat().st_mode & 0o777 == 0o600
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    assert [entry["name"] for entry in record["distributions"]] == [
        "numpy",
        "torch",
    ]
    assert (
        record["distributions"][0]["approved_artifact_sha256"]
        == hashlib.sha256(numpy.read_bytes()).hexdigest()
    )
    assert record["distributions"][0]["installed_content_sha256"] == "c" * 64
    with pytest.raises(FileExistsError):
        build_environment_lock(
            artifacts=[numpy.resolve(), torch.resolve()], output=output.resolve()
        )


@pytest.mark.parametrize(
    ("installed", "artifacts", "match"),
    (
        ({"numpy": "2.2.6", "torch": "2.7.0"}, (("numpy", "2.2.6"),), "lack reviewed"),
        ({"numpy": "2.2.6"}, (("numpy", "2.2.6"), ("torch", "2.7.0")), "not installed"),
        ({"numpy": "2.2.6"}, (("numpy", "2.1.0"),), "versions differ"),
        ({"numpy": "2.2.6"}, (("numpy", "2.2.6"), ("NumPy", "2.2.6")), "duplicate"),
    ),
)
def test_builder_rejects_inventory_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    installed: dict[str, str],
    artifacts: tuple[tuple[str, str], ...],
    match: str,
) -> None:
    paths = [
        _wheel(tmp_path, name=name, version=version, filename=f"artifact-{index}.whl")
        for index, (name, version) in enumerate(artifacts)
    ]
    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.build_environment_lock._installed_distribution_inventory",
        lambda: {name: (version, "c" * 64) for name, version in installed.items()},
    )

    with pytest.raises(ValueError, match=match):
        build_environment_lock(
            artifacts=[path.resolve() for path in paths],
            output=(tmp_path / "environment-lock.json").resolve(),
        )


def test_builder_rejects_relative_symlink_and_non_wheel_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.build_environment_lock._installed_distribution_inventory",
        lambda: {"numpy": ("2.2.6", "c" * 64)},
    )
    wheel = _wheel(tmp_path, name="numpy", version="2.2.6")
    with pytest.raises(ValueError, match="absolute"):
        build_environment_lock(
            artifacts=[Path(wheel.name)],
            output=(tmp_path / "environment-lock.json").resolve(),
        )

    link = tmp_path / "linked.whl"
    link.symlink_to(wheel)
    with pytest.raises(ValueError, match="non-symlink"):
        build_environment_lock(
            artifacts=[link],
            output=(tmp_path / "environment-lock.json").resolve(),
        )

    archive = tmp_path / "numpy.zip"
    archive.write_bytes(wheel.read_bytes())
    with pytest.raises(ValueError, match="wheel files"):
        build_environment_lock(
            artifacts=[archive],
            output=(tmp_path / "environment-lock.json").resolve(),
        )


def test_builder_rejects_conflicting_wheel_metadata_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "renamed.whl"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr(
            "other-2.2.6.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: numpy\nVersion: 2.2.6\n",
        )
    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.build_environment_lock._installed_distribution_inventory",
        lambda: {"numpy": ("2.2.6", "c" * 64)},
    )

    with pytest.raises(ValueError, match="differs from METADATA"):
        build_environment_lock(
            artifacts=[artifact.resolve()],
            output=(tmp_path / "environment-lock.json").resolve(),
        )
