"""Keep frozen public source identities stable across physical file moves.

The vocabulary ledger stays immutable. Only definitions explicitly listed in the
migration ledger are attributed to their historical public owner; unknown names
and additional parameters remain visible to the ordinary census.
"""

from __future__ import annotations

import ast
import json
from functools import lru_cache
from pathlib import Path

_MANIFEST = Path("docs/development/migration/independent-structure-moves.json")


def _module(path: str) -> str:
    parts = Path(path).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


@lru_cache(maxsize=8)
def _identities(payload: bytes) -> dict[tuple[str, str], str]:
    rows = json.loads(payload)["moves"]
    identities: dict[tuple[str, str], str] = {}
    historical: dict[tuple[str, str], str] = {}
    for row in rows:
        old, new = row["old"], row["new"]
        if not old.endswith(".py"):
            continue
        for name in row["symbols"]:
            if name.startswith("__"):
                continue
            key, origin = (new, name), (old, name)
            if key in identities and identities[key] != old:
                raise ValueError(f"ambiguous relocated source: {key}")
            if origin in historical and historical[origin] != new:
                raise ValueError(f"duplicate implementation owner: {origin}")
            identities[key] = old
            historical[origin] = new
        if not row.get("partial"):
            identities.setdefault((new, "<module>"), old)
    return identities


def source_identity(package_root: Path, path: str, qualname: str) -> str:
    """Return the historical owner of one explicitly relocated definition."""
    manifest = package_root.parent / _MANIFEST
    if not manifest.is_file():
        return path
    mapping = _identities(manifest.read_bytes())
    return mapping.get((path, qualname.split(".", 1)[0]), path)


def source_path(repository_root: Path, historical_path: str) -> Path:
    """Resolve one whole-file relocation without changing its evidence identifier."""
    manifest = repository_root / _MANIFEST
    if manifest.is_file():
        matches = {
            row["new"]
            for row in json.loads(manifest.read_bytes())["moves"]
            if row["old"] == historical_path and not row.get("partial")
        }
        if len(matches) == 1:
            return repository_root / matches.pop()
    return repository_root / historical_path


def source_location(repository_root: Path, identifier: str) -> Path:
    """Locate the current implementation of a historical ledger identifier."""
    path, _, tail = identifier.partition("::")
    name = tail.split("::", 1)[0].split(".", 1)[0]
    manifest = repository_root / _MANIFEST
    if manifest.is_file():
        for (current, symbol), previous in _identities(manifest.read_bytes()).items():
            if previous == path and symbol == name:
                return repository_root / current
    return repository_root / path


def relocation_errors(repository_root: Path) -> tuple[str, ...]:
    """Reject missing owners, ambiguous mappings, and retained implementations."""
    manifest = repository_root / _MANIFEST
    if not manifest.is_file():
        return ()
    payload = manifest.read_bytes()
    try:
        _identities(payload)
    except ValueError as exc:
        return (str(exc),)
    errors: list[str] = []
    for row in json.loads(payload)["moves"]:
        old, new = row["old"], row["new"]
        if not old.endswith(".py"):
            continue
        if any(
            Path(path).is_absolute() or ".." in Path(path).parts for path in (old, new)
        ):
            errors.append(f"relocation path must stay in the repository: {old}")
            continue
        current, previous = repository_root / new, repository_root / old
        if not current.is_file():
            errors.append(f"missing relocated implementation: {new}")
            continue
        current_tree = ast.parse(current.read_text(encoding="utf-8"))
        declared = {
            node.name
            for node in ast.walk(current_tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if not set(row["symbols"]) <= declared:
            errors.append(f"relocation claims absent definitions: {new}")
        if row.get("compatibility") == "none":
            if previous.exists():
                errors.append(f"removed entry point still exists: {old}")
            continue
        if _module(old) == _module(new):
            if previous.is_file():
                errors.append(
                    f"module-to-package move retained its old implementation: {old}"
                )
            continue
        if not previous.is_file():
            errors.append(f"missing historical import facade: {old}")
            continue
        tree = ast.parse(previous.read_text(encoding="utf-8"))
        actual = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        duplicated = (set(row["symbols"]) - {"__getattr__"}) & actual
        if duplicated:
            errors.append(
                f"historical facade still defines implementations: {old}: {sorted(duplicated)}"
            )
        target = _module(new)
        imports = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == target
            for alias in node.names
        }
        delegates = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"import_module", "_import_module"}
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == target
            for node in ast.walk(tree)
        )
        if not delegates and not (set(row["symbols"]) - {"__getattr__"}) <= imports:
            errors.append(
                f"historical facade does not forward its declared symbols: {old}"
            )
    return tuple(errors)
