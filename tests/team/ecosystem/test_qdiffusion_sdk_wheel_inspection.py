from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from examples.qdiffusion_kaiwu import inspect_sdk_wheel as inspection_module
from examples.qdiffusion_kaiwu.inspect_sdk_wheel import (
    build_sdk_wheel_inspection,
    inspect_sdk_wheel,
)

pytestmark = pytest.mark.unit

_FILENAME = "kaiwu-1.3.1-cp310-none-manylinux1_x86_64.whl"


def _wheel(
    root: Path,
    *,
    name: str = "kaiwu",
    version: str = "1.3.1",
    member_name: str = "kaiwu/__init__.py",
    license_header: str | None = None,
) -> Path:
    path = root / _FILENAME
    dist_info = f"{name}-{version}.dist-info"
    license_line = f"License-Expression: {license_header}\n" if license_header else ""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{dist_info}/METADATA",
            "Metadata-Version: 2.4\n"
            f"Name: {name}\n"
            f"Version: {version}\n"
            "Author: Qboson Inc\n"
            "Requires-Python: >=3.10\n"
            "Requires-Dist: numpy==2.2.6\n"
            f"{license_line}",
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\n"
            "Root-Is-Purelib: true\n"
            "Tag: cp310-none-manylinux1_x86_64\n",
        )
        archive.writestr(member_name, b"")
        archive.writestr("kaiwu/license/license_settings.so", b"compiled")
    path.chmod(0o600)
    return path


def _accept_fixture_digest(monkeypatch: pytest.MonkeyPatch, wheel: Path) -> None:
    monkeypatch.setitem(
        inspection_module.EXPECTED_IDENTITY,
        "sha256",
        hashlib.sha256(wheel.read_bytes()).hexdigest(),
    )


def test_inspection_records_identity_without_granting_permission(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wheel = _wheel(tmp_path)
    _accept_fixture_digest(monkeypatch, wheel)
    output = tmp_path / "inspection.json"

    digest = inspect_sdk_wheel(
        wheel_path=wheel.resolve(), output_path=output.resolve()
    )

    record = json.loads(output.read_text(encoding="utf-8"))
    assert set(record) == {
        "archive_member_count",
        "author",
        "classification",
        "compiled_extension_count",
        "dependencies",
        "distribution",
        "execution_approved",
        "license_file_paths",
        "license_metadata_present",
        "provider_use_approved",
        "redistribution_approved",
        "requires_python",
        "root_is_purelib",
        "schema",
        "sdk_version",
        "sha256",
        "size_bytes",
        "source_url",
        "static_inspection_only",
        "version",
        "wheel_filename",
        "wheel_tags",
    }
    assert record["classification"] == "artifact_identity_verified_unapproved"
    assert record["distribution"] == "kaiwu"
    assert record["sdk_version"] == "1.3.1"
    assert record["dependencies"] == ["numpy==2.2.6"]
    assert record["compiled_extension_count"] == 1
    assert record["license_metadata_present"] is False
    assert record["license_file_paths"] == []
    assert record["static_inspection_only"] is True
    assert record["execution_approved"] is False
    assert record["provider_use_approved"] is False
    assert record["redistribution_approved"] is False
    assert output.stat().st_mode & 0o777 == 0o600
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        inspect_sdk_wheel(wheel_path=wheel.resolve(), output_path=output.resolve())


def test_inspection_reports_license_evidence_without_interpreting_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wheel = _wheel(
        tmp_path,
        member_name="kaiwu-1.3.1.dist-info/licenses/LICENSE.txt",
        license_header="LicenseRef-Provider",
    )
    _accept_fixture_digest(monkeypatch, wheel)

    record = build_sdk_wheel_inspection(wheel_path=wheel.resolve())

    assert record["license_metadata_present"] is True
    assert record["license_file_paths"] == [
        "kaiwu-1.3.1.dist-info/licenses/LICENSE.txt"
    ]
    assert record["execution_approved"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("digest", "SHA-256 differs"),
        ("name", "distribution differs"),
        ("version", "version differs"),
        ("unsafe_member", "unsafe member name"),
    ),
)
def test_inspection_rejects_identity_or_archive_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    wheel = _wheel(
        tmp_path,
        name="other" if mutation == "name" else "kaiwu",
        version="1.4.1" if mutation == "version" else "1.3.1",
        member_name="../escape" if mutation == "unsafe_member" else "kaiwu/a.py",
    )
    if mutation != "digest":
        _accept_fixture_digest(monkeypatch, wheel)

    with pytest.raises(ValueError, match=message):
        build_sdk_wheel_inspection(wheel_path=wheel.resolve())


def test_inspection_requires_private_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wheel = _wheel(tmp_path)
    _accept_fixture_digest(monkeypatch, wheel)
    wheel.chmod(0o644)

    with pytest.raises(ValueError, match="accessible by group or others"):
        build_sdk_wheel_inspection(wheel_path=wheel.resolve())
