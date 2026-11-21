# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 check/pytest-changed-files.

"""Select existing tests associated by filename with changed Python sources."""

from pathlib import Path

from tools.development import process
from tools.development.environment import PreparedEnvironment


def comparison_revision(root: Path, revision: str | None) -> str:
    """Resolve an explicit commit or the original ordered main-branch defaults."""
    candidates = [revision] if revision else ["upstream/main", "origin/main", "main"]
    for candidate in candidates:
        result = process.run(
            ["git", "cat-file", "-t", candidate],
            cwd=root,
            capture_output=True,
            check=False,
            timeout=15,
            log_run_to_stderr=False,
        )
        if result.returncode == 0 and result.stdout.strip() == "commit":
            return candidate
    if revision:
        raise ValueError(f"No revision {revision!r}")
    raise ValueError("No default revision found to compare against")


def changed_test_paths(root: Path, revision: str) -> list[str]:
    """Map changed files to existing local or native centralized test files.

    This filename-based selection does not infer transitive dependencies.
    Deleted test files are excluded. Results are sorted and deduplicated.
    """
    environment = PreparedEnvironment(None, revision, str(root))
    tests: set[str] = set()
    for name in environment.get_changed_files():
        path = Path(name)
        if path.suffix != ".py":
            continue
        stem = path.stem
        source_stem = stem if stem.endswith("_test") else stem + "_test"
        candidates = [path.with_name(source_stem + ".py")]
        native_stem = stem if stem.startswith("test_") else "test_" + stem
        candidates.append(path.with_name(native_stem + ".py"))
        if path.parts[0] in {"flagquantum", "tools", "examples"}:
            candidates.append(Path("tests/unit") / (native_stem + ".py"))
        for candidate in candidates:
            if (root / candidate).is_file():
                tests.add(candidate.as_posix())
    return sorted(tests)
