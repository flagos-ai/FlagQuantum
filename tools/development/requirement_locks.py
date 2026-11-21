# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 dev_tools/requirements/re-pip-compile.sh.

"""Compile a consistent development lock and its constrained sub-environments."""

import argparse
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

from tools.development import process
from tools.development.files import WriteIfDifferent

# The input directory is supplied explicitly by the environment's owner.
PROFILES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dev", ("dev-tools", "runtime")),
    ("runtime", ("runtime",)),
    ("format", ("runtime", "format")),
    ("pylint", ("runtime", "pylint")),
    ("pytest", ("runtime", "pytest")),
    ("mypy", ("runtime", "mypy")),
    ("docs", ("runtime", "docs")),
    ("pip-tools", ("pip-tools",)),
)


def compile_locks(
    source_root: Path,
    output_root: Path,
    *,
    executable: str = "pip-compile",
) -> int:
    """Compile all eight profiles, publishing only after every command succeeds.

    source_root contains the owner's deps directory. Relative requirement and
    constraint includes retain their paths inside the copied input tree. Output
    files are written through the existing equal-content writer. A write failure
    can leave a prefix of updated locks, so callers may retry the whole command.
    Existing destination locks participate in resolution, as in the source.
    """
    source_root = source_root.resolve(strict=True)
    output_root = output_root.resolve()
    required = {name for _, names in PROFILES for name in names}
    for name in sorted(required):
        if not (source_root / "deps" / f"{name}.txt").is_file():
            raise FileNotFoundError(f"Missing requirement profile: deps/{name}.txt")
    with tempfile.TemporaryDirectory(prefix="fq-requirement-locks-") as directory:
        staged = Path(directory)
        shutil.copytree(source_root / "deps", staged / "deps")
        environments = staged / "envs"
        environments.mkdir()
        for profile, _ in PROFILES:
            previous = output_root / f"{profile}.env.txt"
            if previous.is_file():
                shutil.copy2(previous, environments / previous.name)
        for index, (profile, names) in enumerate(PROFILES):
            arguments = [
                executable,
                f"--output-file=envs/{profile}.env.txt",
                "--resolver=backtracking",
            ]
            if index:
                arguments.append("--constraint=envs/dev.env.txt")
            arguments.extend(f"deps/{name}.txt" for name in names)
            result = process.run(arguments, cwd=staged, check=False, timeout=1200)
            if result.returncode:
                return int(result.returncode)
        # Read every result before publishing any, including missing-output errors.
        results = {
            profile: (environments / f"{profile}.env.txt").read_text(encoding="utf-8")
            for profile, _ in PROFILES
        }
        output_root.mkdir(parents=True, exist_ok=True)
        for profile, content in results.items():
            with WriteIfDifferent(output_root / f"{profile}.env.txt") as destination:
                destination.write(content)
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pip-compile", default="pip-compile")
    arguments = parser.parse_args(argv)
    raise SystemExit(
        compile_locks(
            arguments.source_root,
            arguments.output,
            executable=arguments.pip_compile,
        )
    )


if __name__ == "__main__":
    main()
