"""Inspect the pinned Kaiwu wheel without importing or executing it."""

from __future__ import annotations

import argparse
import hashlib
import io
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any

from examples.qdiffusion_kaiwu.private_io import (
    read_private_bytes,
    write_private_json_exclusive,
)
from examples.qdiffusion_kaiwu.sdk_approval import EXPECTED_IDENTITY

SCHEMA = "flagquantum.qboson_kaiwu_sdk_wheel_inspection"
VERSION = "1.0"

_MAX_WHEEL_BYTES = 32 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 4096
_MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_MAX_METADATA_BYTES = 1024 * 1024
_MAX_WHEEL_METADATA_BYTES = 64 * 1024
_MAX_HEADER_LENGTH = 1024


def _safe_member_name(name: str) -> bool:
    if not name or name.startswith("/") or "\\" in name or "\x00" in name:
        return False
    return all(part not in ("", ".", "..") for part in name.rstrip("/").split("/"))


def _single_dist_info_member(
    members: list[zipfile.ZipInfo], filename: str
) -> zipfile.ZipInfo:
    matches = [
        member
        for member in members
        if not member.is_dir()
        and len(PurePosixPath(member.filename).parts) == 2
        and PurePosixPath(member.filename).parts[0].endswith(".dist-info")
        and PurePosixPath(member.filename).name == filename
    ]
    if len(matches) != 1:
        raise ValueError(f"wheel must contain exactly one dist-info/{filename}")
    return matches[0]


def _bounded_header(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > _MAX_HEADER_LENGTH
        or not value.isprintable()
    ):
        raise ValueError(f"wheel {label} is missing or invalid")
    return value.strip()


def _bounded_headers(values: list[str] | None, *, label: str) -> list[str]:
    if values is None:
        return []
    if len(values) > 256:
        raise ValueError(f"wheel {label} has too many values")
    return sorted(_bounded_header(value, label=label) for value in values)


def _license_paths(members: list[zipfile.ZipInfo]) -> list[str]:
    paths: list[str] = []
    for member in members:
        if member.is_dir():
            continue
        path = PurePosixPath(member.filename)
        basename = path.name.lower()
        parents = {part.lower() for part in path.parts[:-1]}
        if (
            basename in {"license", "copying", "notice"}
            or any(
                basename.startswith(f"{prefix}{separator}")
                for prefix in ("license", "copying", "notice")
                for separator in (".", "-")
            )
            or "licenses" in parents
        ):
            paths.append(member.filename)
    return sorted(paths)


def build_sdk_wheel_inspection(*, wheel_path: Path) -> dict[str, Any]:
    """Return identity-only evidence for the pinned, unapproved Kaiwu wheel."""

    if wheel_path.name != EXPECTED_IDENTITY["wheel_filename"]:
        raise ValueError("wheel filename differs from the pinned 1.3.1 artifact")
    encoded = read_private_bytes(
        wheel_path, label="Kaiwu SDK review wheel", max_bytes=_MAX_WHEEL_BYTES
    )
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != EXPECTED_IDENTITY["sha256"]:
        raise ValueError("wheel SHA-256 differs from the pinned 1.3.1 artifact")
    try:
        with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
            members = archive.infolist()
            if not members or len(members) > _MAX_ARCHIVE_MEMBERS:
                raise ValueError("wheel archive member count is outside the bound")
            names = [member.filename for member in members]
            if len(names) != len(set(names)):
                raise ValueError("wheel archive contains duplicate member names")
            if not all(_safe_member_name(name) for name in names):
                raise ValueError("wheel archive contains an unsafe member name")
            if any(member.flag_bits & 0x1 for member in members):
                raise ValueError("wheel archive contains an encrypted member")
            total_size = sum(member.file_size for member in members)
            if total_size > _MAX_UNCOMPRESSED_BYTES:
                raise ValueError("wheel uncompressed contents exceed the bound")
            metadata_member = _single_dist_info_member(members, "METADATA")
            wheel_member = _single_dist_info_member(members, "WHEEL")
            if metadata_member.file_size > _MAX_METADATA_BYTES:
                raise ValueError("wheel METADATA exceeds the bounded size")
            if wheel_member.file_size > _MAX_WHEEL_METADATA_BYTES:
                raise ValueError("wheel WHEEL metadata exceeds the bounded size")
            metadata = BytesParser(policy=policy.default).parsebytes(
                archive.read(metadata_member)
            )
            wheel_metadata = BytesParser(policy=policy.default).parsebytes(
                archive.read(wheel_member)
            )
    except zipfile.BadZipFile:
        raise ValueError("wheel is not a valid ZIP archive") from None

    distribution = _bounded_header(metadata.get("Name"), label="Name")
    sdk_version = _bounded_header(metadata.get("Version"), label="Version")
    if distribution.casefold() != EXPECTED_IDENTITY["distribution"]:
        raise ValueError("wheel distribution differs from the pinned identity")
    if sdk_version != EXPECTED_IDENTITY["sdk_version"]:
        raise ValueError("wheel version differs from the pinned identity")
    author = _bounded_header(metadata.get("Author"), label="Author")
    requires_python = _bounded_header(
        metadata.get("Requires-Python"), label="Requires-Python"
    )
    dependencies = _bounded_headers(
        metadata.get_all("Requires-Dist"), label="Requires-Dist"
    )
    tags = _bounded_headers(wheel_metadata.get_all("Tag"), label="Tag")
    if not tags:
        raise ValueError("wheel has no compatibility tag")
    purelib = _bounded_header(
        wheel_metadata.get("Root-Is-Purelib"), label="Root-Is-Purelib"
    ).casefold()
    if purelib not in {"true", "false"}:
        raise ValueError("wheel Root-Is-Purelib is invalid")
    license_headers = [
        *metadata.get_all("License", []),
        *metadata.get_all("License-Expression", []),
        *metadata.get_all("License-File", []),
    ]
    license_metadata_present = any(
        isinstance(value, str) and bool(value.strip()) for value in license_headers
    )
    license_paths = _license_paths(members)
    compiled_extensions = sum(
        PurePosixPath(member.filename).suffix.lower() in {".so", ".pyd", ".dylib"}
        for member in members
        if not member.is_dir()
    )
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "classification": "artifact_identity_verified_unapproved",
        "distribution": distribution,
        "sdk_version": sdk_version,
        "wheel_filename": wheel_path.name,
        "source_url": EXPECTED_IDENTITY["source_url"],
        "sha256": digest,
        "size_bytes": len(encoded),
        "author": author,
        "requires_python": requires_python,
        "dependencies": dependencies,
        "wheel_tags": tags,
        "root_is_purelib": purelib == "true",
        "archive_member_count": len(members),
        "compiled_extension_count": compiled_extensions,
        "license_metadata_present": license_metadata_present,
        "license_file_paths": license_paths,
        "static_inspection_only": True,
        "execution_approved": False,
        "provider_use_approved": False,
        "redistribution_approved": False,
    }


def inspect_sdk_wheel(*, wheel_path: Path, output_path: Path) -> str:
    """Write a private static-inspection record and return its SHA-256."""

    record = build_sdk_wheel_inspection(wheel_path=wheel_path)
    write_private_json_exclusive(output_path, record)
    encoded = read_private_bytes(
        output_path, label="Kaiwu SDK wheel inspection", max_bytes=128 * 1024
    )
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        digest = inspect_sdk_wheel(
            wheel_path=arguments.wheel,
            output_path=arguments.output,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Private static SDK inspection written: sha256:{digest}")


if __name__ == "__main__":
    main()
