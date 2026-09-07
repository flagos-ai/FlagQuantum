"""Fail when built distributions contain repository-only or generated files."""

from __future__ import annotations

import argparse
import email
import tarfile
import zipfile
from pathlib import Path

FORBIDDEN_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".fq_jax_cache",
    "tests",
    "benchmarks",
    "examples",
    "datasets",
}
FORBIDDEN_SUFFIXES = {
    ".pyc",
    ".parquet",
    ".ipynb",
    ".onnx",
    ".ckpt",
    ".safetensors",
    ".pt",
    ".pth",
}
MAX_DISTRIBUTION_BYTES = 5_000_000
REQUIRED_MEMBER_SUFFIXES = (
    "flagquantum/simulation/numerics/double-single-contract.toml",
    "flagquantum/runtime/profiles/split_real_imag_statevector_p0.json",
    "flagquantum/runtime/profiles/split_real_imag_statevector_p1.json",
    "flagquantum/runtime/profiles/split_real_imag_statevector_p2_precision.json",
    "flagquantum/runtime/profiles/split_real_imag_statevector_p3_double_single.json",
    "flagquantum/runtime/profiles/split_real_imag_statevector_p4_device_double_single.json",
    "flagquantum/runtime/profiles/statevector_local_p0.json",
)


def _members(path: Path) -> tuple[str, ...]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return tuple(archive.namelist())
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            return tuple(member.name for member in archive.getmembers())
    return ()


def _forbidden(member: str) -> bool:
    path = Path(member)
    return (
        bool(FORBIDDEN_PARTS.intersection(path.parts))
        or path.suffix in FORBIDDEN_SUFFIXES
    )


def _missing_required_members(members: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        suffix
        for suffix in REQUIRED_MEMBER_SUFFIXES
        if not any(member.endswith(suffix) for member in members)
    )


def _wheel_metadata(path: Path) -> tuple[str, ...]:
    with zipfile.ZipFile(path) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        message = email.message_from_bytes(archive.read(metadata_name))
    return tuple(message.get_all("Requires-Dist", ()))


def artifact_errors(path: Path) -> tuple[str, ...]:
    members = _members(path)
    errors = [f"forbidden member: {name}" for name in members if _forbidden(name)]
    errors.extend(
        f"required package resource is missing: {name}"
        for name in _missing_required_members(members)
    )
    if path.stat().st_size > MAX_DISTRIBUTION_BYTES:
        errors.append(f"artifact exceeds {MAX_DISTRIBUTION_BYTES} bytes")
    if not any(Path(name).name.startswith("LICENSE") for name in members):
        errors.append("license file is missing")
    if path.suffix == ".whl":
        invalid_roots = [
            name
            for name in members
            if not (name.startswith("flagquantum/") or ".dist-info/" in name)
        ]
        errors.extend(f"unexpected wheel member: {name}" for name in invalid_roots)
        core_requirements = tuple(
            requirement
            for requirement in _wheel_metadata(path)
            if "extra ==" not in requirement
        )
        if len(core_requirements) != 1 or not core_requirements[0].lower().startswith(
            "torch"
        ):
            errors.append(
                f"core dependencies must contain only torch, got {core_requirements}"
            )
    return tuple(errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    artifacts = tuple(sorted(args.directory.glob("*.whl"))) + tuple(
        sorted(args.directory.glob("*.tar.gz"))
    )
    if not artifacts:
        parser.error(f"no wheel or sdist found in {args.directory}")
    violations = {path.name: artifact_errors(path) for path in artifacts}
    violations = {name: members for name, members in violations.items() if members}
    if violations:
        for name, members in violations.items():
            print(f"{name}: artifact errors: {members}")
        return 1
    print(f"verified {len(artifacts)} distribution artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
