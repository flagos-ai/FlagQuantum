# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 dev_tools/build-protos.sh.

"""Generate Python protobuf modules and typing stubs in an explicit directory."""

import argparse
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from tools.development import process


def _generated_files(root: Path) -> dict[Path, Path]:
    return {
        path.relative_to(root): path
        for path in sorted(root.rglob("*_pb2.py*"))
        if path.is_file() and path.suffix in {".py", ".pyi"}
    }


def build_protobuf(
    source_root: Path,
    output_root: Path,
    schemas: Sequence[Path],
    *,
    check: bool = False,
) -> int:
    """Compile explicit schemas, or compare generated modules and stubs.

    Schemas are relative to source_root. Both modes generate into a temporary
    directory first. Compiler failures return their exact status without copying
    partial outputs. Check mode compares all generated modules in output_root.
    Stale files are reported by check mode and retained by generation mode.
    """
    source_root = source_root.resolve(strict=True)
    output_root = output_root.resolve()
    if not source_root.is_dir():
        raise ValueError("Schema root must be a directory")
    if not schemas:
        raise ValueError("At least one protobuf schema is required")
    relative_schemas = []
    for schema in schemas:
        path = (source_root / schema).resolve(strict=True)
        if not path.is_relative_to(source_root) or path.suffix != ".proto":
            raise ValueError("Schemas must be .proto files inside the schema root")
        if not path.is_file():
            raise ValueError("Schema must be a regular file")
        relative_schemas.append(path.relative_to(source_root).as_posix())
    with tempfile.TemporaryDirectory(prefix="fq-protobuf-") as directory:
        generated_root = Path(directory)
        result = process.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                "-I=.",
                f"--python_out={generated_root}",
                f"--mypy_out={generated_root}",
                *relative_schemas,
            ],
            cwd=source_root,
            check=False,
            timeout=1200,
            log_run_to_stderr=False,
        )
        if result.returncode:
            return int(result.returncode)
        generated = _generated_files(generated_root)
        if not generated:
            print("Protobuf compiler produced no Python modules or typing stubs")
            return 1
        existing = _generated_files(output_root)
        differences = sorted(
            path
            for path in generated.keys() | existing.keys()
            if path not in generated
            or path not in existing
            or generated[path].read_bytes() != existing[path].read_bytes()
        )
        if check:
            for path in differences:
                print(f"Generated protobuf differs: {path.as_posix()}")
            if differences:
                return 1
            print("Protobuf-generated Python modules and typing stubs are up to date.")
            return 0
        output_root.mkdir(parents=True, exist_ok=True)
        for path in sorted(generated.keys() & set(differences)):
            destination = output_root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Replace each complete artifact atomically on the destination device.
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=".fq-protobuf-", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                shutil.copy2(generated[path], temporary_path)
                temporary_path.replace(destination)
            finally:
                temporary_path.unlink(missing_ok=True)
        return 0


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("schemas", type=Path, nargs="+")
    arguments = parser.parse_args(argv)
    try:
        result = build_protobuf(
            arguments.schema_root,
            arguments.output,
            arguments.schemas,
            check=arguments.check,
        )
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    else:
        raise SystemExit(result)


if __name__ == "__main__":
    main()
