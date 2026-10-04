"""Count the wire-named vocabulary the package exposes to its users.

Six surfaces are scanned, and the reason each needs its own walk is stated once
here rather than rediscovered per slice:

- **Parameters** (`census`) are the migration ledger: every `wire`-named
  parameter on the public function surface is a site a slice must retire.
- **Attribute names** (`attribute_census`) are the same rename reached through a
  different door: `result.wires` is on the user's screen but is not a parameter,
  so the parameter ledger cannot see it. An attribute is a *persisted key* rather
  than a name when its spelling reaches a payload, which is decided structurally
  below. Three declaration kinds count: a `field` (annotation or assignment in the
  class body), a `member` (a property or method, so `fq.Circuit.n_wires` is on the
  ledger too), and an `instance` attribute (`self.name = ...` inside a method, so
  `TextDrawer().wire_order` is on the ledger too).
- **Definition names** (`definition_census`) are the third door: a module-level
  public function or class whose own name contains `wire`, such as
  `flagquantum/simulation/pauli.py::infer_n_wires_from_dense_state`. Renaming the
  parameter and leaving the name would not finish the migration.
- **Message strings** (`message_strings`) are reported and never ledgered.
- **Documentation** (`documentation_census`) is the fifth door that reaches the
  user without passing through Python at all. It is not the same question as
  `message_strings`: a literal in a refusal sentence is a runtime string, while a
  keyword argument written into a Markdown file is an instruction to the reader,
  and only the second one can be wrong in a way the reader discovers by copying
  it. The matcher is therefore narrow on purpose -- it counts *keyword arguments*
  (`n_wires=22`) and not the bare word, because a paragraph that says
  "CUDA-Q's wire zero is least-significant" is quoting someone else's vocabulary
  and rewording it would make the comparison false.
- **Call sites** (`qubit_keyword_mismatches`) are the door the other five cannot
  see, because they all read declarations. A ledger can record that
  `observable_wire` left a signature; it cannot record the caller somewhere else in
  the package that still writes `observable_wire=`, and Python raises nothing until
  that branch runs. So this surface asks a different kind of question -- not "how
  much is left" but "does the defect exist" -- and its answer has to be nothing.

Both decisions that make the parameter count reproducible were measured wrong
first, so they are stated here once:

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
- **A dunder is public.** `_is_private_name` treats `__init__` as a user entry
  point, because `fq.Circuit(n_wires=3)` is the single most user-visible call in
  the package. An underscore-only rule filed every public constructor under
  private code and hid twelve canonical sites and three aliases.

A wire-named parameter is a **deprecated alias** exactly when the same signature
also declares its qubit-named replacement (`wire` next to `qubit`, `wires` next
to `qubits`). That is machine-decidable and matches how
`flagquantum/core/_qubit_aliases.py` implements the alias, so the ledger never
has to be re-tabulated by inspection.

Deciding the same question for an attribute was measured wrong first as well, and
the correction matters because it reverses the direction of the error. The first
rule was the natural one -- an attribute whose spelling appears as a string
literal is a payload key -- and it was reported as finding **zero** of the 102
attribute names. That number is an artifact of the ruler, not a fact about the
package: the census wrote identifiers as `path::Qualname.::attr` and the
classifier split on the last `.`, so it searched the literal table for
`"::n_wires"`, a spelling that cannot occur.

Split on the census separator instead, and the same rule over-reaches badly: it
finds 20 of the 21 payload keys, but it also marks 85 of the 102 names as
persisted, which would freeze 65 attributes that carry no payload obligation at
all. Both failure modes are disqualifying, and the first one hid the second.

The rule implemented in `_payload_index` is therefore structural rather than
textual: an attribute's spelling is a payload key when its owning class declares
a serialization method or calls `asdict(self)` (`serialization_method`), when the
class is a field type of such a class so `asdict` recurses into it (`contained`),
or when the class is named inside such a method's body (`nested`).

That structural rule is correct for a **field** and wrong for a **member** or an
**instance attribute**, which is why the kinds are separated and carry different
witnesses. `asdict` walks a dataclass's fields, so it can never turn a property's
name into a key: `KrausChannel.n_wires` and `MPSState.n_wires` were both marked
persisted by the class-level rule although neither spelling can leave the process,
and `MPSState` is not even a dataclass. An instance attribute is in the same
position -- `TextDrawer().wire_order` is not a field of a dataclass either. So for
both of those kinds the spelling is a payload name only when one of the *owning
class's own serialization methods* contains it as a string literal
(`payload_literal`), because that method's contract is the cross-process one. Under
that rule exactly one member is excluded, `RuntimePolicy.observable_wires`, whose
`from_dict` still accepts the legacy key -- and `Circuit.n_wires`, whose spelling
appears in the in-process `circuit_param` dict, is correctly *not* excluded, since
no serializer of `Circuit` emits it.

A definition name needs no exclusion rule at all. It is reachable only as a Python
identifier, and the four places the package spells one as a literal were checked by
hand: `kernels/catalog/implementations.py` and `kernels/triton/*.py` declare it as
a source-level catalog `symbol` (the stable kernel identity is the `FQKI-*` id),
and `simulation/native_cpu/__init__.py` and `simulation/pauli.py` list it in
`__all__`. The same rename updates all four.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import io
import re
import subprocess
import sys
import tokenize
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACKAGE_ROOT = REPOSITORY_ROOT / "flagquantum"

PUBLIC_SURFACE = "canonical"
DEPRECATED_ALIAS = "deprecated_alias"
PRIVATE_CODE = "internal"

LEDGERED_ATTRIBUTE = "canonical"
PERSISTED_ATTRIBUTE = "serialized_key"

DECLARED_FIELD = "field"
DECLARED_MEMBER = "member"
DECLARED_INSTANCE = "instance"
DECLARATION_KINDS = (DECLARED_FIELD, DECLARED_MEMBER, DECLARED_INSTANCE)

SERIALIZATION_METHODS = frozenset(
    {
        "as_dict",
        "asdict",
        "deserialize",
        "dump",
        "from_dict",
        "from_json",
        "from_payload",
        "from_record",
        "from_toml",
        "load",
        "serialize",
        "to_dict",
        "to_json",
        "to_payload",
        "to_record",
        "to_toml",
        "to_plan_dict",
        "from_plan_dict",
    }
)


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


@dataclass(frozen=True)
class AttributeSite:
    """One wire-named class attribute on the public surface."""

    path: str
    owner: str
    attribute: str
    declaration: str
    verdict: str
    evidence: str
    witness: str

    @property
    def identifier(self) -> str:
        """A ledger key in the same shape as a parameter site's."""

        return f"{self.path}::{self.owner}::{self.attribute}"

    @property
    def replacement(self) -> str:
        return replacement_name(self.attribute)

    @property
    def reason(self) -> str:
        """Why the spelling cannot move yet, phrased so a reader can check it."""

        if not self.witness:
            return ""
        if self.evidence == "payload_literal":
            return (
                f"`{self.witness}` reads the spelling as an input key, so "
                f"`{self.attribute}` is still part of the `{self.owner}` payload "
                "contract"
            )
        if self.evidence == "serialization_method":
            return (
                f"`{self.witness}` builds its payload from its own field names, so "
                f"`{self.attribute}` is a key in the `{self.witness}` payload"
            )
        return (
            f"`{self.witness}` builds its payload with `dataclasses.asdict`, which "
            f"recurses into this class, so `{self.attribute}` is a key in the "
            f"`{self.witness}` payload"
        )

    @property
    def removal_condition(self) -> str:
        """The event that retires the exclusion; never a date and never 'later'."""

        if not self.witness:
            return ""
        if self.evidence == "payload_literal":
            return f"the `{self.owner}` payload schema is versioned and migrated"
        if self.witness == "CircuitIR":
            return "`IR_VERSION` is raised and the payload migrates in the same release"
        return f"the `{self.witness}` payload schema is versioned and migrated"


@dataclass(frozen=True)
class DefinitionSite:
    """One wire-named module-level public function or class."""

    path: str
    name: str
    kind: str

    @property
    def identifier(self) -> str:
        return f"{self.path}::{self.name}"

    @property
    def replacement(self) -> str:
        return replacement_name(self.name)


@dataclass(frozen=True)
class DefinitionCensus:
    """Every wire-named module-level public definition name."""

    ledgered: tuple[DefinitionSite, ...]

    def by_file(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        grouped: dict[str, list[str]] = {}
        for site in self.ledgered:
            grouped.setdefault(site.path, []).append(site.identifier)
        return tuple(
            (path, tuple(sorted(entries))) for path, entries in sorted(grouped.items())
        )


@dataclass(frozen=True)
class AttributeCensus:
    """Every wire-named attribute, split by whether its spelling is a payload key."""

    ledgered: tuple[AttributeSite, ...]
    excluded: tuple[AttributeSite, ...]

    def by_file(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        grouped: dict[str, list[str]] = {}
        for site in self.ledgered:
            grouped.setdefault(site.path, []).append(site.identifier)
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
) -> list[tuple[str, str, str]]:
    """Public class names with their owner, as (owner, name, declaration kind).

    Three doors count, because a class can put a name on the user's screen in
    three ways and each needs a different check. A **field** is an annotation or
    an assignment in the class body. A **member** is a property or method
    definition, because `fq.Circuit.n_wires` is reachable by every user even
    though no assignment declares it. An **instance attribute** is a
    `self.name = ...` assignment inside a method: `TextDrawer().wire_order` and
    `ShardedMPSState().n_wires` are read by callers, and neither of the other two
    doors sees them.
    """

    found: list[tuple[str, str, str]] = []
    for statement in body:
        if isinstance(statement, ast.ClassDef):
            nested_private = private or _is_private_name(statement.name)
            nested_prefix = f"{prefix}{statement.name}."
            seen: set[tuple[str, str]] = set()
            for member in statement.body:
                targets: list[ast.expr] = []
                if isinstance(member, ast.AnnAssign):
                    targets = [member.target]
                elif isinstance(member, ast.Assign):
                    targets = list(member.targets)
                for target in targets:
                    if isinstance(target, ast.Name) and not _is_private_name(target.id):
                        if (
                            not nested_private
                            and (target.id, DECLARED_FIELD) not in seen
                        ):
                            seen.add((target.id, DECLARED_FIELD))
                            found.append((nested_prefix, target.id, DECLARED_FIELD))
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not nested_private and not _is_private_name(member.name):
                        key = (member.name, DECLARED_MEMBER)
                        if key not in seen:
                            seen.add(key)
                            found.append((nested_prefix, member.name, DECLARED_MEMBER))
                    if not nested_private:
                        for node in ast.walk(member):
                            assigned = _self_assigned_name(node)
                            if assigned is None or _is_private_name(assigned):
                                continue
                            key = (assigned, DECLARED_INSTANCE)
                            if key not in seen:
                                seen.add(key)
                                found.append(
                                    (nested_prefix, assigned, DECLARED_INSTANCE)
                                )
            found.extend(
                _declared_attributes(statement.body, nested_prefix, nested_private)
            )
        else:
            found.extend(
                _declared_attributes(getattr(statement, "body", []), prefix, private)
            )
    return found


def _self_assigned_name(node: ast.AST) -> str | None:
    """The name in a `self.name = ...` assignment, if that is what `node` is."""

    if not isinstance(node, ast.Attribute) or not isinstance(node.ctx, ast.Store):
        return None
    if not isinstance(node.value, ast.Name) or node.value.id != "self":
        return None
    return node.attr


def _declared_definitions(
    body: list[ast.stmt], prefix: str, private: bool
) -> list[tuple[str, str, str]]:
    """Public module-level function and class names, as (qualname, name, kind).

    Only names declared at module scope count: a definition nested inside a
    private class or function is not reachable without going through that owner,
    which the attribute surface already covers.
    """

    found: list[tuple[str, str, str]] = []
    for statement in body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "class" if isinstance(statement, ast.ClassDef) else "function"
            if not private and not _is_private_name(statement.name):
                found.append((f"{prefix}{statement.name}", statement.name, kind))
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


def _attribute_sites(root: Path) -> list[tuple[str, str, str, str]]:
    """Every public class name, as (module, owner, name, declaration kind).

    One walk shared by the total, the forbidden-namespace check, and the split, so
    the three cannot disagree about what the surface contains or how it is spelled.
    """

    found: list[tuple[str, str, str, str]] = []
    for path in sorted(root.rglob("*.py")):
        if not is_public_module(path, root):
            continue
        relative = path.relative_to(root.parent).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        for qualname, name, declaration in _declared_attributes(tree.body, "", False):
            found.append((relative, qualname.rstrip("."), name, declaration))
    return found


def public_attribute_names(package_root: Path | None = None) -> tuple[str, ...]:
    """Public class attributes and members that say wire.

    A second surface: the same rename has to reach `MeasurementResult.wires`,
    `ExecutionPlan.shardable_wires`, and `Circuit.n_wires`, and the parameter
    ledger cannot see any of them. `attribute_census` splits this list into the
    names a slice may rename and the spellings that are payload keys; this function
    stays as the total, so the split can be reconciled against it.
    """

    return tuple(
        identifier
        for identifier in all_public_attribute_names(package_root)
        if "wire" in identifier.rsplit("::", 1)[1]
    )


def all_public_attribute_names(package_root: Path | None = None) -> tuple[str, ...]:
    """Every public class name, wire-named or not.

    The forbidden-alternative rule needs the whole namespace: `qubit_indices`
    contains no `wire`, so a scan that looks for `wire` never sees the spelling a
    rule forbids.
    """

    root = DEFAULT_PACKAGE_ROOT if package_root is None else package_root
    return tuple(
        sorted(
            f"{path}::{owner}::{name}"
            for path, owner, name, _ in _attribute_sites(root)
        )
    )


def _definition_sites(root: Path) -> list[tuple[str, str, str]]:
    """Every module-level public definition, as (module, name, kind)."""

    found: list[tuple[str, str, str]] = []
    for path in sorted(root.rglob("*.py")):
        if not is_public_module(path, root):
            continue
        relative = path.relative_to(root.parent).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        for _, name, kind in _declared_definitions(tree.body, "", False):
            found.append((relative, name, kind))
    return found


def definition_census(package_root: Path | None = None) -> DefinitionCensus:
    """Every wire-named module-level public function or class name.

    Renaming a parameter without renaming the function that declares it leaves the
    spelling on the user's screen, and no other surface sees it: `infer_n_wires_from_dense_state`
    is a name, not a parameter and not a class attribute. No name is excluded --
    see the module docstring for the four literal sites that were checked by hand.
    """

    root = DEFAULT_PACKAGE_ROOT if package_root is None else package_root
    ledgered = [
        DefinitionSite(path, name, kind)
        for path, name, kind in _definition_sites(root)
        if "wire" in name
    ]
    return DefinitionCensus(tuple(ledgered))


@dataclass(frozen=True)
class _PayloadIndex:
    """Why an attribute's spelling reaches a payload, and which class proves it.

    The witness is not decoration. An exclusion has to be argued for by name
    ("a key of the `ExecutionPlan` payload"), not by a rule ("looks serialized"),
    because the reader has to be able to check it and to know what event would
    retire it.
    """

    witnesses: dict[str, tuple[str, str]]

    def evidence(self, owner: str) -> tuple[str, str]:
        """The kind of evidence for `owner` and the class that witnesses it."""

        return self.witnesses.get(owner, ("", ""))


def _annotation_names(annotation: ast.expr) -> tuple[str, ...]:
    """Every identifier an annotation mentions, including inside a string form."""

    names: list[str] = []
    for part in ast.walk(annotation):
        if isinstance(part, ast.Name):
            names.append(part.id)
        elif isinstance(part, ast.Constant) and isinstance(part.value, str):
            names.extend(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", part.value))
    return tuple(names)


def _is_serialization_method(member: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether a class member is one of the class's own payload entry points.

    A method counts when its name is a known serializer, or when it calls
    `asdict(self)`, which builds a payload keyed by the class's field names.
    """

    if member.name in SERIALIZATION_METHODS:
        return True
    for call in ast.walk(member):
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "asdict"
        ):
            continue
        if any(
            isinstance(argument, ast.Name) and argument.id == "self"
            for argument in call.args
        ):
            return True
    return False


def _payload_index(root: Path) -> _PayloadIndex:
    """Build the structural evidence for which attribute spellings become keys.

    This exists because the obvious rule does not work: a scan for string
    literals finds zero of the attribute names, since `dataclasses.asdict` turns
    each field name into a key without any literal appearing in the source.
    """

    trees: dict[str, ast.Module] = {}
    for path in sorted(root.rglob("*.py")):
        if is_public_module(path, root):
            trees[path.relative_to(root.parent).as_posix()] = ast.parse(
                path.read_text(encoding="utf-8"), filename=path.name
            )

    serializing: set[str] = set()
    nested: dict[str, str] = {}
    field_types: dict[str, set[str]] = {}
    for tree in trees.values():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            field_types.setdefault(node.name, set())
            for member in node.body:
                if isinstance(member, ast.AnnAssign):
                    field_types[node.name].update(_annotation_names(member.annotation))
                elif isinstance(member, ast.Assign) and member.value is not None:
                    field_types[node.name].update(_annotation_names(member.value))
                if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if _is_serialization_method(member):
                    serializing.add(node.name)
                for call in ast.walk(member):
                    if not (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Name)
                        and call.func.id == "asdict"
                    ):
                        continue
                    for argument in (*call.args, *(kw.value for kw in call.keywords)):
                        if argument is None:
                            continue
                        for part in ast.walk(argument):
                            if isinstance(part, ast.Name) and part.id != "self":
                                nested.setdefault(part.id, node.name)

    witnesses: dict[str, tuple[str, str]] = {}
    for witness in sorted(serializing):
        witnesses.setdefault(witness, ("serialization_method", witness))
    for witness in sorted(serializing):
        for owner in sorted(field_types.get(witness, ())):
            if owner != witness:
                witnesses.setdefault(owner, ("contained", witness))
    for owner, witness in sorted(nested.items()):
        witnesses.setdefault(owner, ("nested", witness))
    return _PayloadIndex(witnesses)


def _member_literal_index(root: Path) -> dict[str, str]:
    """Which member or instance spellings a class's own serialization methods use.

    A member needs its own evidence rule because the field rule is wrong for it:
    `dataclasses.asdict` walks fields, so it can never emit a property's name, and
    applying the class-level `serialization_method` witness to members excluded
    `KrausChannel.n_wires` and `MPSState.n_wires`, neither of which can leave the
    process. An instance attribute (`TextDrawer().wire_order`) is in the same
    position for the same reason: it is not a dataclass field. Scoping the literal
    scan to the owning class's serialization methods is what makes it exact:
    `RuntimePolicy.from_dict` accepts `observable_wires`, so that spelling is an
    input key, while `Circuit.circuit_param` -- the one case that made a
    class-wide literal scan look attractive -- is an in-process dict that no
    serializer emits.
    """

    index: dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        if not is_public_module(path, root):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for member in node.body:
                if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if not _is_serialization_method(member):
                    continue
                for part in ast.walk(member):
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        index.setdefault(
                            f"{node.name}::{part.value}", f"{node.name}.{member.name}"
                        )
    return index


def attribute_census(package_root: Path | None = None) -> AttributeCensus:
    """Split wire-named attributes by whether their spelling is a payload key.

    The split is the whole point. A `wire`-named attribute is either a name the
    migration will rename or a spelling that already leaves the process as a
    serialized key, and the cheap way to tell them apart -- look for the name as
    a string literal -- gets it wrong in both directions: one payload key here is
    produced by `dataclasses.asdict` and therefore has no literal to find, and 65
    names that are merely attribute accesses inherit a literal from an unrelated
    use of the same word. See the module docstring for the measurement.
    """

    root = DEFAULT_PACKAGE_ROOT if package_root is None else package_root
    index = _payload_index(root)
    ledgered: list[AttributeSite] = []
    excluded: list[AttributeSite] = []
    literals = _member_literal_index(root)
    for path, owner, attribute, declaration in _attribute_sites(root):
        if "wire" not in attribute:
            continue
        if declaration in (DECLARED_MEMBER, DECLARED_INSTANCE):
            witness = literals.get(f"{owner}::{attribute}", "")
            evidence = "payload_literal" if witness else ""
        else:
            evidence, witness = index.evidence(owner)
        site = AttributeSite(
            path,
            owner,
            attribute,
            declaration,
            PERSISTED_ATTRIBUTE if evidence else LEDGERED_ATTRIBUTE,
            evidence,
            witness,
        )
        (excluded if evidence else ledgered).append(site)
    return AttributeCensus(tuple(ledgered), tuple(excluded))


DOCUMENTATION_EXCLUDED_PREFIXES = (
    "benchmarks/results/",
    "docs/api-changes/",
    "docs/development/",
)
DOCUMENTATION_EXCLUDED_FILES = ("docs/reference/QUBIT_NAMING_MIGRATION.md",)

# Reported as its own bucket by `prose_noun_census`, and *not* excluded from the
# keyword scan. An architecture decision record states what was true when it was
# accepted, so its prose is left alone like the other dated records; but it is not
# an instruction to a reader, so it stays inside the census's file list and a future
# ADR that writes out a `wires=` example still fails the gate. Counting the two
# separately is what makes both facts visible at once instead of forcing one choice.
PROSE_SCANNED_ONLY_PREFIXES = ("docs/architecture/decisions/",)

# A keyword argument, with the lookbehind that keeps an attribute read
# (`info.wires=`) and the text of a CLI flag (`--n-wires 12`) out. The name is
# filtered after the match rather than encoded in the pattern, because a wire-named
# keyword can carry `wire` anywhere in the identifier -- `wires`, `n_wires`,
# `terminal_wires`, `show_wire_labels` -- and a pattern that required the literal
# substring to start the name missed `wires=(0,)` entirely.
#
# A backtick is deliberately *not* in the lookbehind. It was, and the exclusion hid
# the case the surface exists to find: a document that writes `` (`wires=`) `` in a
# code span is naming the keyword it tells the reader to pass, exactly as a code
# block is, and the one occurrence the backtick rule hid was the one that had gone
# stale. A doc that writes `` `n_wires=2` `` while describing a live deprecated alias
# is the other side of that coin, and it is recorded as an exemption rather than
# silenced by the pattern.
KEYWORD_ARGUMENT = re.compile(r"(?<![\w.-])(?P<name>[A-Za-z_]\w*)\s*=(?!=)")


@dataclass(frozen=True)
class DocumentationCensus:
    """Wire-named keyword arguments written into current documentation.

    ``sites`` holds one entry per *occurrence* so that a count is preserved
    without pinning a line number: reflowing a paragraph moves every line in it,
    and a ledger keyed on lines would report a failure for an edit that changed no
    vocabulary at all.
    """

    sites: tuple[str, ...]

    def counts(self) -> Counter[str]:
        return Counter(self.sites)


def documentation_files(repository_root: Path | None = None) -> tuple[str, ...]:
    """Tracked Markdown that documents the current tree, as repository paths.

    The exclusion is by *kind of document*, not by whether a file happens to be
    free of the word today. A dated record under ``docs/api-changes`` or
    ``docs/development`` states what the tree said on its date, so it must be able
    to quote the retired spelling; rewriting those would destroy the evidence the
    migration rests on. The naming reference is the same case: naming both
    spellings is its entire job. ``benchmarks/results`` is raw recorded evidence --
    the source of a run that produced a number -- and ``AGENTS.md`` forbids
    rewriting recorded evidence, so a stale spelling there is a fact about a past
    measurement rather than an instruction to a reader.
    """

    root = REPOSITORY_ROOT if repository_root is None else repository_root
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
        timeout=60,
    ).stdout
    return tuple(
        name
        for name in sorted(set(listed.decode("utf-8").split("\0")))
        if name.endswith(".md")
        and name not in DOCUMENTATION_EXCLUDED_FILES
        and not name.startswith(DOCUMENTATION_EXCLUDED_PREFIXES)
    )


def documentation_keywords(text: str) -> tuple[str, ...]:
    """The wire-named keyword arguments one document tells a reader to write.

    Separated from the walk so the matcher can be pinned on its own: the rule that
    matters is *which* spellings count, and that rule is about a line of Markdown,
    not about which files exist.
    """

    return tuple(
        match.group("name") + "="
        for match in KEYWORD_ARGUMENT.finditer(text)
        if "wire" in match.group("name")
    )


def documentation_census(
    repository_root: Path | None = None,
) -> DocumentationCensus:
    """Count the wire-named keyword arguments the documentation tells users to write."""

    root = REPOSITORY_ROOT if repository_root is None else repository_root
    sites: list[str] = []
    for relative in documentation_files(root):
        text = (root / relative).read_text(encoding="utf-8")
        sites.extend(
            f"{relative}::{keyword}" for keyword in documentation_keywords(text)
        )
    return DocumentationCensus(sites=tuple(sorted(sites)))


def prose_noun_census(
    repository_root: Path | None = None,
) -> tuple[tuple[str, int, int], ...]:
    """Occurrences of the bare noun in every tracked Markdown file, by bucket.

    The companion to `documentation_census`, and it exists because the two numbers
    get confused with each other. That one counts *keyword arguments* -- the only
    spelling a reader can copy into a script and have raise -- and is therefore the
    one a gate can enforce. This one counts the bare noun, which no gate can
    enforce, because the documentation has to be able to say that CUDA-Q orders wire
    zero as least-significant and to describe a `--n-wires` flag.

    So a number from here is a *reading*, reported and never asserted, and it is
    reported over **every** tracked Markdown file rather than the census's in-scope
    subset. That is deliberate: the excluded buckets are the bulk of the count and a
    reader comparing this figure against the keyword count needs both halves visible
    in one output rather than one of them silently subtracted.

    Each row is ``(bucket, occurrences, files containing at least one)``, because the
    two together are what let a reader tell "this bucket is untouched" from "this
    bucket has fewer occurrences in more files".

    This replaces a shell loop over directories that reported a plausible number and
    was wrong, because a loop reading `git ls-files` inside a process substitution
    keeps only its final iteration. It was wrong by a factor of two and nothing about
    the output said so, which is the whole argument for measuring it here instead.
    """

    root = REPOSITORY_ROOT if repository_root is None else repository_root
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
        timeout=60,
    ).stdout
    totals: Counter[str] = Counter()
    files: Counter[str] = Counter()
    for name in sorted(set(listed.decode("utf-8").split("\0"))):
        if not name.endswith(".md"):
            continue
        text = (root / name).read_text(encoding="utf-8", errors="replace")
        count = text.lower().count("wire")
        bucket = "everything else"
        for prefix in (
            *DOCUMENTATION_EXCLUDED_PREFIXES,
            *PROSE_SCANNED_ONLY_PREFIXES,
        ):
            if name.startswith(prefix):
                bucket = prefix
                break
        else:
            if name in DOCUMENTATION_EXCLUDED_FILES:
                bucket = name
        totals[bucket] += count
        if count:
            files[bucket] += 1
    order = (
        *DOCUMENTATION_EXCLUDED_PREFIXES,
        *DOCUMENTATION_EXCLUDED_FILES,
        *PROSE_SCANNED_ONLY_PREFIXES,
        "everything else",
    )
    return tuple((bucket, totals[bucket], files[bucket]) for bucket in order)


def _import_aliases(tree: ast.Module, package: str) -> dict[str, str]:
    """Map a local name to a resolvable dotted path, over *every* import in a file.

    Function-local imports are the normal style for a heavy or optional dependency,
    so walking only the module body would miss exactly the call sites that matter
    most -- a distributed executor imported inside a branch, for instance.
    """

    resolved: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            base = package
            for _ in range(max(node.level - 1, 0)):
                base = base.rpartition(".")[0]
            for alias in node.names:
                if alias.name == "*":
                    continue
                resolved[alias.asname or alias.name] = (
                    f"{base}.{node.module}.{alias.name}"
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                resolved[alias.asname or alias.name.split(".")[0]] = alias.name
    return resolved


def _callee_path(node: ast.expr, aliases: dict[str, str]) -> str | None:
    """Resolve a call target, or `None` when the callee cannot be named statically."""

    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        base = aliases.get(node.value.id)
        if base is None:
            return None
        return f"{base}.{node.attr}"
    return None


def _accepted_keywords(target_path: str) -> tuple[str, ...] | None:
    """The keyword names a callee accepts, or `None` when it cannot be decided.

    `None` covers three honest refusals: the module will not import (an optional
    accelerator is absent), the attribute is not a function, or the function takes
    `**kwargs`, in which case *any* keyword has to be assumed accepted. A gate must
    skip an undecidable site rather than guess, so that a reported mismatch is
    always a real one.
    """

    module_path, _, attribute = target_path.rpartition(".")
    if not module_path:
        return None
    try:
        module = importlib.import_module(module_path)
        target = getattr(module, attribute, None)
    except Exception:
        return None
    if target is None or not callable(target):
        return None
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return None
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return None
    return tuple(signature.parameters)


def qubit_keyword_mismatches(package_root: Path | None = None) -> tuple[str, ...]:
    """Call sites that pass a wire- or qubit-named keyword the callee rejects.

    This is the defect class the other five surfaces are structurally blind to. A
    ledger records *declarations*, so `observable_wire` leaving a signature is a
    retirement the gate can see; the caller thirty files away that still writes
    `observable_wire=` is not a site any ledger holds, and nothing in the package's
    own imports fails until that branch actually runs. Rewording the declaration is
    therefore not the last step of a rename -- finding every caller is.

    Both spellings are checked, because the same edit can lag in either direction:
    a caller can be left on the retired `wire` spelling, or written against a
    `qubit` spelling that has not landed. Only keywords carrying the vocabulary are
    reported, so this stays the migration's own question and does not become a
    general unused-keyword audit.
    """

    root = DEFAULT_PACKAGE_ROOT if package_root is None else package_root
    findings: list[str] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root.parent).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        # A relative import resolves against the *containing* package. For a
        # regular module that is its dotted name minus the last component; for an
        # `__init__.py` the module's dotted name already *is* the package, so the
        # extra rpartition there would resolve `flagquantum.a.__init__`'s `from .b`
        # as `flagquantum.b` instead of `flagquantum.a.b`.
        dotted = relative.removesuffix(".py").replace("/", ".")
        if dotted.endswith(".__init__"):
            package = dotted[: -len(".__init__")]
        else:
            package = dotted.rpartition(".")[0]
        aliases = _import_aliases(tree, package)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target_path = _callee_path(node.func, aliases)
            if target_path is None:
                continue
            candidates = [
                keyword.arg
                for keyword in node.keywords
                if keyword.arg is not None
                and ("wire" in keyword.arg or "qubit" in keyword.arg)
            ]
            if not candidates:
                continue
            accepted = _accepted_keywords(target_path)
            if accepted is None:
                continue
            for name in candidates:
                if name not in accepted:
                    findings.append(
                        f"{relative}:{node.lineno}::{target_path}::{name}"
                        f" -- callee accepts {', '.join(accepted)}"
                    )
    return tuple(sorted(findings))


def message_strings(package_root: Path | None = None) -> tuple[str, ...]:

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


WIRE_TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _wire_tokens(text: str) -> tuple[str, ...]:
    """The wire-bearing identifier-shaped tokens in one line of prose."""

    return tuple(word for word in WIRE_TOKEN.findall(text) if "wire" in word.lower())


def _scope_names(tree: ast.Module) -> tuple[tuple[int, int, str], ...]:
    """Every definition's line span and dotted qualname, innermost last."""

    spans: list[tuple[int, int, str]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                end = getattr(child, "end_lineno", child.lineno)
                spans.append((child.lineno, end, name))
                walk(child, f"{name}.")
            else:
                walk(child, prefix)

    walk(tree, "")
    return tuple(spans)


def _definition_names(tree: ast.Module) -> dict[int, str]:
    """Every definition node's dotted qualname, keyed by the node's identity.

    A docstring is keyed by the dotted name rather than the bare one for the same
    reason a comment is: two classes may each define a method of the same name, and a
    key that merged them would let one container's exemption pay for the other's
    occurrence. Identity rather than position keeps the lookup independent of
    ``ast.walk``'s order.
    """

    names: dict[int, str] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                names[id(child)] = name
                walk(child, f"{name}.")
            else:
                walk(child, prefix)

    walk(tree, "")
    return names


def _enclosing(spans: tuple[tuple[int, int, str], ...], line: int) -> str:
    """The innermost definition containing a line, or `<module>` at file scope."""

    best: tuple[int, str] | None = None
    for start, end, name in spans:
        if start <= line <= end and (best is None or start >= best[0]):
            best = (start, name)
    return "<module>" if best is None else best[1]


def _example_lines(doc: str) -> set[int]:
    """The zero-based docstring lines that lie inside an executable example block.

    ``doctest`` starts an example at a line whose stripped form begins with ``>>>``
    and ends it at the next blank line; the lines between are either the statement or
    the output it is compared against. Both are read by ``doctest`` rather than by a
    person, so neither is prose.

    This distinction is load-bearing. ``tests/api_contract/test_public_docstring_examples.py``
    runs every ``>>>`` the public entries carry, and a wire-named token on one of those
    lines is a *code* token: a rewriting pass that could not tell the two apart turned
    ``Instruction.wires`` into ``.qubits`` in the ``Circuit.compose`` example, and the
    example that had resolved for years started raising ``AttributeError``. The doctest
    runner is what governs those tokens, so they are counted apart from the prose ledger.
    """

    if ">>>" not in doc:
        return set()
    inside = False
    lines: set[int] = set()
    for index, line in enumerate(doc.splitlines()):
        stripped = line.strip()
        if stripped.startswith(">>>"):
            inside = True
        elif not stripped:
            inside = False
        # An output line keeps the block open. A prose line cannot open one, because
        # only `>>>` sets the flag.
        if inside:
            lines.add(index)
    return lines


def _tree_fingerprint(root: Path) -> str:
    """A digest of every byte a docstring scan is about to read.

    The contract gate re-derives the same two docstring surfaces once per assertion a
    test makes, and on this package one derivation is seconds of parsing; a test module
    that makes forty of them spends minutes re-reading a tree that has not changed. The
    memo below is keyed by this digest rather than by the path alone, so a tree that is
    rewritten in-process -- which is exactly what the gate's own negative tests do, into
    a ``tmp_path`` -- cannot be served a scan of its previous contents.

    The digest is over contents, not over timestamps and sizes: 624 files and 7 MB cost
    34 ms to hash here against 1600 ms to parse, and a content digest cannot be fooled
    by a rewrite that happens to restore an identical size and mtime.
    """

    digest = hashlib.blake2b(digest_size=16)
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root.parent).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


_DOCSTRING_SURFACES: dict[
    tuple[str, str],
    tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]],
] = {}


def _docstring_surfaces(
    root: Path,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    """Both docstring surfaces from one pass: the prose, and the executable examples.

    They are read in the same traversal because they are disjoint halves of one
    docstring, decided by ``_example_lines``: an index inside an executable block is an
    example and every other index is prose. Reading them apart would parse every file
    twice and would also mean the split had two chances to disagree with itself.

    The result is memoized per process because the gate asks for it repeatedly and the
    tree does not change between those asks. The key is a content digest, so the memo
    is an optimization and never an answer about a tree other than the one on disk.
    """

    key = (str(root), _tree_fingerprint(root))
    cached = _DOCSTRING_SURFACES.get(key)
    if cached is not None:
        return cached

    prose: list[tuple[str, str]] = []
    examples_found: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root.parent).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
        spans = _scope_names(tree)
        names = _definition_names(tree)
        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                continue
            doc = ast.get_docstring(node, clean=False)
            if not doc or "wire" not in doc.lower():
                continue
            qualname = names.get(id(node), "<module>")
            lines = doc.splitlines()
            examples = _example_lines(doc)
            for index, line in enumerate(lines):
                target = examples_found if index in examples else prose
                for token in _wire_tokens(line):
                    target.append((f"{relative}::{qualname}", token))
        for comment in tokenize.generate_tokens(io.StringIO(source).readline):
            if comment.type != tokenize.COMMENT or "wire" not in comment.string.lower():
                continue
            scope = _enclosing(spans, comment.start[0])
            for word in _wire_tokens(comment.string):
                prose.append((f"{relative}::{scope}", word))

    surfaces = tuple(sorted(prose)), tuple(sorted(examples_found))
    _DOCSTRING_SURFACES[key] = surfaces
    return surfaces


def docstring_census(
    package_root: Path | None = None,
) -> tuple[tuple[str, str], ...]:
    """Every wire-bearing token a package docstring's prose or a comment publishes.

    A docstring is what ``help()`` prints and what an editor shows on hover, and a
    comment is what the next reader of the module is taught, so this surface reaches
    a user exactly as a signature does -- while being invisible to every ledger that
    reads an AST signature. It is the only surface here that is neither a name the
    package declares nor a keyword it documents; it is the package *talking*.

    A site is ``relative::Qualname::token``, and the gate reconciles the whole
    multiset per container, so a second occurrence cannot hide behind the first. The
    scope a comment belongs to is the innermost definition containing it, which keeps
    the key stable when lines move inside that definition.

    An executable example block inside a docstring is skipped, because those lines are
    code that ``doctest`` runs rather than prose a reader reads; they are counted by
    ``docstring_example_tokens`` instead. See ``_example_lines``.
    """

    root = DEFAULT_PACKAGE_ROOT if package_root is None else package_root
    return _docstring_surfaces(root)[0]


def docstring_example_tokens(
    package_root: Path | None = None,
) -> tuple[tuple[str, str], ...]:
    """Every wire-bearing token on a docstring line that ``doctest`` executes.

    Reported rather than reconciled, for the same reason ``message_strings`` is: the
    rule these obey is not a vocabulary rule but the one the doctest runner already
    enforces -- the example has to keep working. A token here is a code token, so
    rewording it is a rename, and a rename needs the authorization a rename needs.
    Counting them apart is what makes that visible *before* a rewriting pass reaches
    them, which is exactly what failed once: ``Instruction.wires`` in
    ``Circuit.compose`` became ``.qubits``, a spelling no attribute carries.
    """

    root = DEFAULT_PACKAGE_ROOT if package_root is None else package_root
    return _docstring_surfaces(root)[1]


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
    attributes = attribute_census()
    fields = sum(1 for site in attributes.ledgered if site.declaration == "field")
    members = sum(1 for site in attributes.ledgered if site.declaration == "member")
    print(
        "attribute surface: "
        f"{len(attributes.ledgered)} canonical names to rename "
        f"({fields} fields and {members} members) and "
        f"{len(attributes.excluded)} persisted keys to leave alone"
    )
    # Each loop below names its own variable. They used to share `site`, which forced
    # mypy to pick the first binding's type and then reject the other three -- and
    # `tools/` is outside the strict type check in CI, so the errors were real but
    # invisible to the gate that would have caught them.
    for excluded in attributes.excluded:
        print(
            f"  persisted  {excluded.identifier}  "
            f"({excluded.evidence} by {excluded.witness}) -> {excluded.replacement}"
        )
    definitions = definition_census()
    print(
        "definition surface: "
        f"{len(definitions.ledgered)} module-level public names to rename"
    )
    for definition in definitions.ledgered:
        print(
            f"  {definition.kind:8} {definition.identifier} -> {definition.replacement}"
        )
    documentation = documentation_census()
    print(
        "documentation surface: "
        f"{len(documentation.sites)} wire-named keyword arguments written into "
        f"{len(documentation_files())} tracked Markdown files"
    )
    for keyword, count in sorted(documentation.counts().items()):
        print(f"  {count:4d}  {keyword}")
    print("reported but not ledgered:")
    print(f"  {len(message_strings()):4d}  string literals")
    # Printed after the enforced surfaces, and labelled as a reading rather than a
    # count, so the two numbers about prose cannot be mistaken for each other. The
    # keyword line above is what a gate can enforce; this one is what a reader can
    # see, over every tracked Markdown file rather than the census's subset.
    rows = prose_noun_census()
    print("prose noun, reported only (occurrences over every tracked Markdown file):")
    for bucket, occurrences, files in rows:
        print(f"  {occurrences:5d}  {files:4d} files  {bucket}")
    print(f"  {sum(row[1] for row in rows):5d}           total")
    mismatches = qubit_keyword_mismatches()
    print(
        "call sites: "
        f"{len(mismatches)} wire- or qubit-named keywords passed to a callee that "
        "rejects them (invariant: 0)"
    )
    for finding in mismatches:
        print(f"  {finding}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
