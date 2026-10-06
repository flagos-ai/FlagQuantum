from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from examples.qdiffusion_kaiwu import build_environment_lock as build_module
from examples.qdiffusion_kaiwu import verify_environment_lock as verify_module
from examples.qdiffusion_kaiwu.build_environment_lock import (
    build_environment_lock,
)
from examples.qdiffusion_kaiwu.verify_environment_lock import (
    _distribution_content_sha256,
    load_environment_lock,
    verify_environment_lock,
    verify_frozen_environment_lock,
)

pytestmark = pytest.mark.unit


def _record() -> dict[str, object]:
    return {
        "schema": "flagquantum.qboson_qdiffusion_environment_lock",
        "version": "1.0",
        "inventory_policy": "exact",
        "python_version": "3.10.16",
        "distributions": [
            {
                "name": "numpy",
                "version": "2.2.6",
                "approved_artifact_sha256": "a" * 64,
                "installed_content_sha256": "c" * 64,
            },
            {
                "name": "torch",
                "version": "2.7.0",
                "approved_artifact_sha256": "b" * 64,
                "installed_content_sha256": "d" * 64,
            },
        ],
    }


def _write(path: Path, record: dict[str, object]) -> None:
    path.write_text(json.dumps(record), encoding="utf-8")
    path.chmod(0o600)


def test_environment_lock_requires_private_exact_sorted_inventory(
    tmp_path: Path,
) -> None:
    path = tmp_path / "environment-lock.json"
    _write(path, _record())

    record, digest = load_environment_lock(path)

    assert record["inventory_policy"] == "exact"
    assert len(digest) == 64

    record["distributions"] = list(reversed(record["distributions"]))  # type: ignore[arg-type]
    _write(path, record)
    with pytest.raises(ValueError, match="unique and sorted"):
        load_environment_lock(path)


def test_environment_lock_rejects_placeholder_and_public_file(tmp_path: Path) -> None:
    path = tmp_path / "environment-lock.json"
    record = _record()
    record["python_version"] = "3.10.<required>"
    _write(path, record)
    with pytest.raises(ValueError, match="exact Python"):
        load_environment_lock(path)

    _write(path, _record())
    path.chmod(0o644)
    with pytest.raises(ValueError, match="group or others"):
        load_environment_lock(path)


@pytest.mark.parametrize("nested", (False, True))
def test_environment_lock_rejects_undeclared_fields(
    tmp_path: Path, nested: bool
) -> None:
    path = tmp_path / "environment-lock.json"
    record = _record()
    target = record["distributions"][0] if nested else record  # type: ignore[index]
    target["sdk_code"] = "must-not-be-retained"  # type: ignore[index]
    _write(path, record)

    with pytest.raises(ValueError, match="field set differs from schema"):
        load_environment_lock(path)


@pytest.mark.parametrize("unsafe_parent", ("public", "symlink"))
def test_environment_lock_rejects_unsafe_parent(
    tmp_path: Path, unsafe_parent: str
) -> None:
    private_parent = tmp_path / "private"
    private_parent.mkdir(mode=0o700)
    path = private_parent / "environment-lock.json"
    _write(path, _record())
    if unsafe_parent == "public":
        private_parent.chmod(0o755)
        candidate = path
    else:
        linked_parent = tmp_path / "linked"
        linked_parent.symlink_to(private_parent, target_is_directory=True)
        candidate = linked_parent / path.name

    with pytest.raises(ValueError, match="parent"):
        load_environment_lock(candidate)


def test_environment_lock_matches_exact_runtime_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "environment-lock.json"
    _write(path, _record())
    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.verify_environment_lock.platform.python_version",
        lambda: "3.10.16",
    )
    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.verify_environment_lock._installed_distribution_inventory",
        lambda: {"numpy": ("2.2.6", "c" * 64), "torch": ("2.7.0", "d" * 64)},
    )

    _, digest = verify_environment_lock(path)

    assert len(digest) == 64

    _, frozen_digest = verify_frozen_environment_lock(
        path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert frozen_digest == digest

    with pytest.raises(ValueError, match="differs from frozen"):
        verify_frozen_environment_lock(path, expected_sha256="f" * 64)


@pytest.mark.parametrize(
    ("inventory", "match"),
    (
        ({"numpy": ("2.2.6", "c" * 64)}, "missing"),
        (
            {
                "numpy": ("2.2.6", "c" * 64),
                "torch": ("2.7.0", "d" * 64),
                "extra": ("1", "e" * 64),
            },
            "unlisted",
        ),
        (
            {"numpy": ("2.2.6", "c" * 64), "torch": ("2.6.0", "d" * 64)},
            "versions differ",
        ),
        (
            {"numpy": ("2.2.6", "0" * 64), "torch": ("2.7.0", "d" * 64)},
            "installed contents differ",
        ),
    ),
)
def test_environment_lock_rejects_runtime_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    inventory: dict[str, tuple[str, str]],
    match: str,
) -> None:
    path = tmp_path / "environment-lock.json"
    _write(path, _record())
    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.verify_environment_lock.platform.python_version",
        lambda: "3.10.16",
    )
    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.verify_environment_lock._installed_distribution_inventory",
        lambda: inventory,
    )

    with pytest.raises(ValueError, match=match):
        verify_environment_lock(path)


def test_distribution_content_digest_binds_record_file_set(tmp_path: Path) -> None:
    first = tmp_path / "package" / "first.py"
    first.parent.mkdir()
    first.write_text("value = 1\n", encoding="utf-8")
    second = tmp_path / "package" / "weights.bin"
    second.write_bytes(b"weights")

    class Distribution:
        def __init__(self) -> None:
            self.files = [Path("package/first.py"), Path("package/weights.bin")]

        @staticmethod
        def locate_file(entry: Path) -> Path:
            return tmp_path / entry

    before = _distribution_content_sha256(Distribution(), name="example")  # type: ignore[arg-type]
    second.write_bytes(b"changed")
    after = _distribution_content_sha256(Distribution(), name="example")  # type: ignore[arg-type]

    assert before != after

    second.unlink()
    second.symlink_to(first)
    with pytest.raises(ValueError, match="missing or unsafe"):
        _distribution_content_sha256(Distribution(), name="example")  # type: ignore[arg-type]


def test_distribution_content_digest_rechecks_complete_file_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "package" / "first.py"
    first.parent.mkdir()
    first.write_text("value = 1\n", encoding="utf-8")
    second = tmp_path / "package" / "second.py"
    second.write_text("value = 2\n", encoding="utf-8")

    class Distribution:
        def __init__(self) -> None:
            self.files = [Path("package/first.py"), Path("package/second.py")]

        @staticmethod
        def locate_file(entry: Path) -> Path:
            return tmp_path / entry

    real_capture = verify_module.capture_regular_file
    calls = 0

    def capture_then_replace(path: Path, *, label: str):
        nonlocal calls
        snapshot = real_capture(path, label=label)
        calls += 1
        if calls == 2:
            first.write_bytes(first.read_bytes())
        return snapshot

    monkeypatch.setattr(verify_module, "capture_regular_file", capture_then_replace)

    with pytest.raises(ValueError, match="changed after identity capture"):
        _distribution_content_sha256(Distribution(), name="example")  # type: ignore[arg-type]


def test_environment_lock_binds_wheel_metadata_and_digest_to_one_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "example-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "example-1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: example\nVersion: 1.0\n",
        )
    original = wheel.read_bytes()

    def inventory_after_replacement() -> dict[str, tuple[str, str]]:
        wheel.write_bytes(original)
        return {"example": ("1.0", "a" * 64)}

    monkeypatch.setattr(
        build_module,
        "_installed_distribution_inventory",
        inventory_after_replacement,
    )
    output = tmp_path / "environment-lock.json"

    with pytest.raises(ValueError, match="changed after identity capture"):
        build_environment_lock(artifacts=[wheel], output=output)

    assert not output.exists()
