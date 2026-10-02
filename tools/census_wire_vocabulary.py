"""Count the wire-named parameters declared by the package's functions.

The count this script prints is the progress ledger for the qubit-vocabulary
migration: `contracts/qubit-vocabulary-contract.toml` records the same sites, a
migration slice moves them from `baseline` to `retired`, and
`tools/check_qubit_vocabulary.py` fails if a baseline site disappears without
being retired or if a wire-named parameter appears outside the ledger.

Both decisions that make the count reproducible were measured wrong first, so
they are stated here once:

- **Every parameter kind counts.** Reading only `posonlyargs + args + kwonlyargs`
  misses `Circuit.any(*wires, unitary=...)` and its deprecated `Circuit.unitary`
  spelling entirely. Variadic names keep their `*` prefix so the two kinds stay
  distinguishable in the ledger.
- **A public namespace is not a private module, and private code is private all
  the way down.** A path component starting with `_` marks private code, with one
  exception: `__init__.py` *is* its package's public namespace, so
  `flagquantum/observables/__init__.py` counts although its filename starts with
  an underscore. Within a module, a name that starts with `_` is private, and so
  is everything nested inside it — an earlier scan reported methods of private
  classes as if they were module-level public functions, which inflated the count
  by 16 sites.

A wire-named parameter is a **deprecated alias** exactly when the same signature
also declares its qubit-named replacement (`wire` next to `qubit`, `wires` next
to `qubits`). That is machine-decidable and matches how
`flagquantum/core/_qubit_aliases.py` implements the alias, so the ledger never
has to be re-tabulated by inspection.
"""

from __future__ import annotations

import ast
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACKAGE_ROOT = REPOSITORY_ROOT / "flagquantum"

PUBLIC_SURFACE = "canonical"
DEPRECATED_ALIAS = "deprecated_alias"
PRIVATE_CODE = "internal"


@dataclass(frozen=True)
class Site:
    """One wire-named parameter on one function."""

    path: str
    qualname: str
    parameter: str
    kind: str

    @property
    def identifier(self) -> str:
        return site_identifier(self.path, self.qualname, self.parameter)

    @property
    def replacement(self) -> str:
        return replacement_name(self.parameter)

    @property
    def file_entry(self) -> str:
        return f"{self.qualname}::{self.parameter}"


@dataclass(frozen=True)
class Census:
    """Every wire-named parameter the scan can see, split by disposition."""

    canonical: tuple[Site, ...]
    aliases: tuple[Site, ...]
    internal: tuple[Site, ...]

    @property
    def declared(self) -> tuple[Site, ...]:
        """The public function surface: canonical spellings and their aliases."""

        return self.canonical + self.aliases

    def counters(self) -> tuple[tuple[str, int], ...]:
        counts = Counter(_subpackage(site.path) for site in self.canonical)
        return tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def by_file(
        self, sites: tuple[Site, ...]
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        grouped: dict[str, list[str]] = {}
        for site in sites:
            grouped.setdefault(site.path, []).append(site.file_entry)
        return tuple(
            (path, tuple(sorted(entries))) for path, entries in sorted(grouped.items())
        )


def site_identifier(path: str, qualname: str, parameter: str) -> str:
    """A ledger key that survives edits above it, unlike a line number."""

    return f"{path}::{qualname}::{parameter}"


def replacement_name(parameter: str) -> str:
    """The qubit-named spelling of a wire-named parameter."""

    if "wires" in parameter:
        return parameter.replace("wires", "qubits")
    return parameter.replace("wire", "qubit")


def _subpackage(path: str) -> str:
    parts = Path(path).parts
    if len(parts) <= 2:
        return "/".join(parts)
    return "/".join(parts[:2])


def supersedes(name: str, replacement: str) -> bool:
    """Whether `name` is the deprecated spelling of `replacement`.

    A name that does not contain `wire` supersedes nothing, so the answer for it
    is `False` rather than the trivially equal replacement.
    """

    if "wire" not in name:
        return False
    return replacement_name(name) == replacement


def is_public_module(path: Path, package_root: Path) -> bool:
    """Whether `path` belongs to the package's public import surface."""

    for component in path.relative_to(package_root).parts:
        if not component.startswith("_"):
            continue
        if component == "__init__.py":
            continue
        return False
    return True


def parameter_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...]:
    """Every declared parameter name, variadic ones spelled with their prefix."""

    named = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
    names = [argument.arg for argument in named]
    for argument, prefix in ((node.args.vararg, "*"), (node.args.kwarg, "**")):
        if argument is not None:
            names.append(f"{prefix}{argument.arg}")
    return tuple(names)


def _is_private_name(name: str) -> bool:
    """Whether a declared name is hidden from the user, not merely underscore-led.

    A leading underscore means private, with one exception: a dunder is Python's
    public protocol, so `Circuit.__init__` is a user entry point and `self._cache`
    is not. Treating every underscore-led name as private would drop the
    constructor -- the single most user-visible signature there is -- into the
    excluded bucket.
    """

    if name.startswith("__") and name.endswith("__"):
        return False
    return name.startswith("_")


def _declared_functions(
    body: list[ast.stmt], prefix: str, private: bool
) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, bool]]:
    found: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, bool]] = []
    for statement in body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nested_private = private or _is_private_name(statement.name)
            found.append((f"{prefix}{statement.name}", statement, nested_private))
            found.extend(
                _declared_functions(
                    statement.body, f"{prefix}{statement.name}.", nested_private
                )
            )
        elif isinstance(statement, ast.ClassDef):
            nested_private = private or _is_private_name(statement.name)
            found.extend(
                _declared_functions(
                    statement.body, f"{prefix}{statement.name}.", nested_private
                )
            )
        else:
            found.extend(
                _declared_functions(getattr(statement, "body", []), prefix, private)
            )
    return found


def _declared_attributes(
    body: list[ast.stmt], prefix: str, private: bool
) -> list[tuple[str, str]]:
    """Class attributes declared by annotation or assignment, with their owner."""

    found: list[tuple[str, str]] = []
    for statement in body:
        if isinstance(statement, ast.ClassDef):
            nested_private = private or _is_private_name(statement.name)
            nested_prefix = f"{prefix}{statement.name}."
            for member in statement.body:
                targets: list[ast.expr] = []
                if isinstance(member, ast.AnnAssign):
                    targets = [member.target]
                elif isinstance(member, ast.Assign):
                    targets = list(member.targets)
                for target in targets:
                    if isinstance(target, ast.Name) and not _is_private_name(target.id):
                        if not nested_private:
                            found.append((nested_prefix, target.id))
            found.extend(
                _declared_attributes(statement.body, nested_prefix, nested_private)
            )
        else:
            found.extend(
                _declared_attributes(getattr(statement, "body", []), prefix, private)
            )
    return found


def census(package_root: Path | None = None) -> Census:
    """Scan the package and split the wire-named parameters by disposition."""

    root = DEFAULT_PACKAGE_ROOT if package_root is None else package_root
    canonical: list[Site] = []
    aliases: list[Site] = []
    internal: list[Site] = []
    for path in sorted(root.rglob("*.py")):
        if not is_public_module(path, root):
            continue
        relative = path.relative_to(root.parent).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        for qualname, node, private in _declared_functions(tree.body, "", False):
            names = parameter_names(node)
            for parameter in names:
                if "wire" not in parameter:
                    continue
                if private:
                    internal.append(Site(relative, qualname, parameter, PRIVATE_CODE))
                elif replacement_name(parameter) in names:
                    aliases.append(
                        Site(relative, qualname, parameter, DEPRECATED_ALIAS)
                    )
                else:
                    canonical.append(
                        Site(relative, qualname, parameter, PUBLIC_SURFACE)
                    )
    return Census(tuple(canonical), tuple(aliases), tuple(internal))


def public_parameter_names(package_root: Path | None = None) -> frozenset[str]:
    """Every parameter name on the public function surface, not only wire ones.

    The forbidden-alternative rule cannot be checked from the wire-named sites:
    `qubit_indices` contains no `wire`, so a scan looking for `wire` never sees
    the name a rule forbids. This walk reports the whole namespace of parameter
    names so the gate can ask about any spelling.
    """

    root = DEFAULT_PACKAGE_ROOT if package_root is None else package_root
    names: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        if not is_public_module(path, root):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        for _, node, private in _declared_functions(tree.body, "", False):
            if not private:
                names.update(parameter_names(node))
    return frozenset(names)


def public_attribute_names(package_root: Path | None = None) -> tuple[str, ...]:
    """Annotated or assigned class attributes on the public surface that say wire.

    A second surface, measured but not ledgered: the same rename has to reach
    `MeasurementResult.wires` and `ExecutionPlan.shardable_wires`, and the
    parameter ledger cannot see either. Reporting the count keeps the gap a
    number rather than an impression, and names the slice that must extend the
    gate before it lands.
    """

    root = DEFAULT_PACKAGE_ROOT if package_root is None else package_root
    found: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if not is_public_module(path, root):
            continue
        relative = path.relative_to(root.parent).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        for qualname, attribute in _declared_attributes(tree.body, "", False):
            if "wire" in attribute:
                found.append(f"{relative}::{qualname}::{attribute}")
    return tuple(sorted(found))


def message_strings(package_root: Path | None = None) -> tuple[str, ...]:
    """String literals on the public surface that say wire.

    Reported, never ledgered. A literal scan cannot tell a serialized key from a
    refusal sentence, so a ledger built on it would fail on legitimate rewording
    while still missing a key built by concatenation.
    """

    root = DEFAULT_PACKAGE_ROOT if package_root is None else package_root
    found: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if not is_public_module(path, root):
            continue
        relative = path.relative_to(root.parent).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "wire" in node.value:
                    found.append(f"{relative}:{node.lineno}")
    return tuple(found)


def main() -> int:
    scanned = census()
    print(
        "wire vocabulary census: "
        f"{len(scanned.canonical)} canonical parameters and "
        f"{len(scanned.aliases)} deprecated aliases on the public function "
        f"surface, {len(scanned.internal)} parameters in private code"
    )
    for subpackage, count in scanned.counters():
        print(f"  {count:4d}  {subpackage}")
    print("deprecated aliases:")
    for site in scanned.aliases:
        print(f"  {site.identifier} -> {site.replacement}")
    print("measured but not ledgered:")
    print(f"  {len(public_attribute_names()):4d}  public attribute names")
    print(f"  {len(message_strings()):4d}  string literals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
