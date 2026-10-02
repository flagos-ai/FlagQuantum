#!/usr/bin/env python3
"""Fail-closed checker for the FlagQuantum QEC <-> CUDA-Q alignment checklist.

The checklist (`contracts/qec-cudaq-alignment-checklist.toml`) is prose turned
into rows. Nothing in it is trusted: every row must prove itself against the
repository it describes.

What this checks, per row
-------------------------
1. `evidence` paths exist inside the repo (file or directory).
2. `symbols_present` resolve to a real definition in the checked tree.
3. `symbols_absent` resolve to nothing anywhere in the checked tree, and every
   `negative_search` path is still missing -- so a row that claims a gap cannot
   survive the gap being filled, and a row that claims alignment cannot survive
   the symbol being deleted.
4. `maturity_ref`, when present, names an existing `[capabilities.*]` entry.
5. `domain_row`, when present, names a capability inside the parity matrix's
   `quantum_error_correction` domain, the row's status does not contradict the
   matrix (an `aligned` checklist row may not sit on an `unsupported` matrix
   row), and the row's maturity entry is one the matrix row names -- through
   `maturity_ref` or through `maturity_refs`, because a matrix row whose
   implementation spans two registry entries names both. A divergence is allowed
   only with a stated `maturity_ref_override_reason`.
6. Status-specific requirements: `aligned` rows need a present symbol, `partial`
   and `absent` rows need a stated `next_action`, `absent` rows need at least
   one `symbols_absent` entry.
7. Every row id is mentioned in the reading document
   (`docs/development/QEC_CUDAQ_ALIGNMENT.md`), so a row cannot exist only in the
   TOML and the prose cannot silently drop one.
8. The header's `checked_by` names a file that exists in this repository, because
   a checklist that points at a tool outside its own tree cannot be gated in CI.

Field-level rows are the same discipline one level down: `equivalent`,
`renamed`, and `reshaped` require the FlagQuantum symbol to exist, and `absent`
requires it not to.

The header is checked too. A row that admits its provenance is unverified
obliges the checklist to state the limit once, and the limit is only worth
stating while the document agrees with the contract that owns it: the recorded
component version, its release line, and the release index it came from must all
be the ones `contracts/cudaq-parity-matrix.toml` records, and a contract that
records no provenance fails the header rather than passing it.

Resolution is source-level (an AST scan), not import-level, on purpose: the
checker must run without torch, without an installed FlagQuantum, and without
the optional `stim` distribution. A symbol resolves when a `def`, `class`, or
module-level assignment of that name exists either in the module the dotted
path names or, for a package path, anywhere in that package -- which is what
re-export through `flagquantum/qec/__init__.py` means in practice.

Exit status is 0 only when every check passes. Failures are printed with the
row id and the reason, so the checklist can be repaired rather than trusted.

Usage
-----
    python3 tools/check_qec_cudaq_alignment.py
    python3 tools/check_qec_cudaq_alignment.py --quiet
    python3 tools/check_qec_cudaq_alignment.py --json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

# Defaults are absolute, so the tool gives the same answer from any working
# directory and a CI step does not have to name the repository it is standing in.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ROOT = str(ROOT)
DEFAULT_CHECKLIST = str(ROOT / "contracts/qec-cudaq-alignment-checklist.toml")
DEFAULT_READING = str(ROOT / "docs/development/QEC_CUDAQ_ALIGNMENT.md")

STATUS_VALUES = ("aligned", "partial", "absent", "out_of_scope")
FIELD_VERDICTS = ("equivalent", "renamed", "reshaped", "absent", "extra")
# A checklist row claiming alignment cannot rest on a matrix row that declares
# the capability unsupported: one of the two documents would be lying.
CONTRADICTIONS = {("aligned", "unsupported"), ("aligned", "out_of_scope")}


@dataclass
class Report:
    """Collected results, so one run reports every problem instead of the first."""

    failures: list[str] = field(default_factory=list)
    checks: int = 0

    def ok(self) -> bool:
        return not self.failures

    def fail(self, where: str, reason: str) -> None:
        self.failures.append(f"{where}: {reason}")

    def check(self) -> None:
        self.checks += 1


# --------------------------------------------------------------------------
# source-level symbol index
# --------------------------------------------------------------------------


class SymbolIndex:
    """Definitions found by parsing every Python file under a root package."""

    def __init__(self, package_root: Path, package_name: str) -> None:
        self.package_root = package_root
        self.package_name = package_name
        # module dotted path -> set of names defined at module top level
        self.modules: dict[str, set[str]] = {}
        # (module, class) -> set of member names
        self.members: dict[tuple[str, str], set[str]] = {}
        # every defined name anywhere, for negative searches
        self.all_names: set[str] = set()
        self.parse_failures: list[str] = []
        self._scan()

    def _module_name(self, path: Path) -> str:
        rel = path.relative_to(self.package_root.parent).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)

    def _scan(self) -> None:
        for path in sorted(self.package_root.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (SyntaxError, UnicodeDecodeError, OSError) as error:
                self.parse_failures.append(f"{path}: {error}")
                continue
            module = self._module_name(path)
            top: set[str] = self.modules.setdefault(module, set())
            for node in tree.body:
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    top.add(node.name)
                    self.all_names.add(node.name)
                    if isinstance(node, ast.ClassDef):
                        members: set[str] = set()
                        for child in node.body:
                            if isinstance(
                                child,
                                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                            ):
                                members.add(child.name)
                            elif isinstance(child, ast.AnnAssign) and isinstance(
                                child.target, ast.Name
                            ):
                                # A dataclass field is a name a caller may read,
                                # so a field-level claim has to be able to prove it.
                                members.add(child.target.id)
                            elif isinstance(child, ast.Assign):
                                members.update(
                                    target.id
                                    for target in child.targets
                                    if isinstance(target, ast.Name)
                                )
                        self.members[(module, node.name)] = members
                        self.all_names.update(members)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            top.add(target.id)
                            self.all_names.add(target.id)
                elif isinstance(node, ast.AnnAssign) and isinstance(
                    node.target, ast.Name
                ):
                    top.add(node.target.id)
                    self.all_names.add(node.target.id)

    def _package_modules(self, dotted: str) -> list[str]:
        prefix = f"{dotted}."
        return [
            name for name in self.modules if name == dotted or name.startswith(prefix)
        ]

    def resolves(self, dotted: str) -> bool:
        """True when `dotted` names a definition the scanned tree really has."""

        if not dotted.startswith(f"{self.package_name}."):
            return False
        parts = dotted.split(".")
        # Longest prefix that names a module or package we know about.
        for cut in range(len(parts) - 1, 0, -1):
            head = ".".join(parts[:cut])
            tail = parts[cut:]
            if not self._package_modules(head):
                continue
            if len(tail) == 1:
                name = tail[0]
                if any(
                    name in self.modules.get(mod, ())
                    for mod in self._package_modules(head)
                ):
                    return True
                continue
            if len(tail) == 2:
                owner, member = tail
                for mod in self._package_modules(head):
                    if owner in self.modules.get(
                        mod, ()
                    ) and member in self.members.get((mod, owner), set()):
                        return True
        return False


# --------------------------------------------------------------------------
# contract readers
# --------------------------------------------------------------------------


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def qec_matrix_rows(
    contracts_dir: Path,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Capability rows and the recorded CUDA-Q surface of the QEC domain."""

    path = contracts_dir / "cudaq-parity-matrix.toml"
    if not path.is_file():
        return {}, []
    data = load_toml(path)
    for domain in data.get("domains", []):
        if domain.get("id") == "quantum_error_correction":
            rows = {}
            for capability in domain.get("capabilities", []):
                if "id" in capability:
                    rows[capability["id"]] = capability
            return rows, list(domain.get("cudaq_surface", []))
    return {}, []


def contract_version_provenance(contracts_dir: Path) -> dict[str, Any]:
    """The parity contract's record of the QEC component's release line.

    The checklist carries the same limit in its header, and the limit is only
    worth stating while the two documents agree about it. Reading it from the
    contract rather than restating it here means the header cannot claim a
    provenance the contract no longer supports.
    """
    path = contracts_dir / "cudaq-parity-matrix.toml"
    if not path.is_file():
        return {}
    return dict(load_toml(path).get("baseline", {}).get("version_provenance", {}))


def maturity_entries(repo_root: Path) -> set[str]:
    path = repo_root / "capability-maturity.toml"
    if not path.is_file():
        return set()
    return set(load_toml(path).get("capabilities", {}).keys())


def matrix_maturity_refs(capability: dict[str, Any]) -> list[str]:
    """Return the registry entries a parity matrix row names.

    A matrix row names one entry through `maturity_ref`, or several through
    `maturity_refs` when its implementation genuinely spans more than one
    registry entry. Both forms are read, so a checklist row is measured against
    every entry the matrix row claims rather than against whichever form was
    written.
    """
    multiple = capability.get("maturity_refs")
    if multiple is not None:
        return list(multiple)
    single = capability.get("maturity_ref")
    return [] if single is None else [single]


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


def check_row(
    row: dict[str, Any],
    index: int,
    repo_root: Path,
    index_of: SymbolIndex,
    matrix: dict[str, dict[str, Any]],
    domain_surface: list[str],
    maturity: set[str],
    report: Report,
) -> dict[str, Any]:
    row_id = row.get("id") or f"row[{index}]"
    where = f"row {row_id}"
    report.check()

    status = row.get("status")
    if status not in STATUS_VALUES:
        report.fail(where, f"status {status!r} is not one of {STATUS_VALUES}")

    for key in (
        "baseline",
        "target",
        "fail_closed",
        "next_action" if status in ("partial", "absent") else None,
    ):
        if key and not str(row.get(key, "")).strip():
            report.fail(where, f"{key} is empty")

    present = row.get("symbols_present", []) or []
    absent = row.get("symbols_absent", []) or []
    negative = row.get("negative_search", []) or []

    if status == "aligned" and not present:
        report.fail(
            where, "an `aligned` row must name at least one symbols_present entry"
        )
    if status == "partial" and not present:
        report.fail(
            where,
            "a `partial` row must name the symbols_present entry that is the partial half",
        )
    if status in ("absent", "out_of_scope") and not (absent or negative):
        report.fail(
            where,
            f"a {status!r} row must prove the gap with symbols_absent or negative_search",
        )

    for symbol in present:
        report.check()
        if not index_of.resolves(symbol):
            report.fail(
                where, f"symbols_present {symbol!r} does not resolve in the repository"
            )
    for symbol in absent:
        report.check()
        if index_of.resolves(symbol):
            report.fail(
                where,
                f"symbols_absent {symbol!r} now resolves -- the gap closed, so this row is stale",
            )
        if symbol in index_of.all_names:
            report.fail(
                where, f"symbols_absent {symbol!r} exists as a definition somewhere"
            )

    for entry in negative:
        report.check()
        kind, _, target = entry.partition(":")
        if kind == "symbol":
            if not target:
                report.fail(where, f"negative_search {entry!r} names no symbol")
            elif index_of.resolves(target):
                report.fail(
                    where,
                    f"negative_search {entry!r} now resolves -- the absence this row records is over",
                )
        else:
            if (repo_root / entry).exists():
                report.fail(
                    where,
                    f"negative_search {entry!r} now exists -- the absence this row records is over",
                )

    for item in row.get("domain_items", []) or []:
        report.check()
        recorded = list(domain_surface)
        # A row that names a capability may also quote that capability's own
        # `cudaq` text: the domain surface list is the coarse inventory, and the
        # per-capability line is the same baseline stated one level down.
        named_row = row.get("domain_row")
        if named_row and named_row in matrix:
            recorded.append(str(matrix[named_row].get("cudaq", "")))
        if not any(item in surface for surface in recorded if surface):
            report.fail(
                where,
                f"domain_items {item!r} is not a recorded CUDA-Q surface item of the "
                "quantum_error_correction domain",
            )

    for evidence in row.get("evidence", []) or []:
        report.check()
        if not (repo_root / evidence).exists():
            report.fail(where, f"evidence path {evidence!r} does not exist")

    maturity_ref = row.get("maturity_ref")
    if maturity_ref:
        report.check()
        if maturity_ref not in maturity:
            report.fail(
                where,
                f"maturity_ref {maturity_ref!r} is not a capability-maturity entry",
            )

    domain_row = row.get("domain_row")
    matrix_status = None
    floor = row.get("floor")
    if (
        floor
        and not domain_row
        and not str(row.get("floor_override_reason", "")).strip()
    ):
        # A priority claim that no matrix row backs is unverifiable arithmetic.
        # Say why the floor stands on its own, or name the row it comes from.
        report.fail(
            where,
            "states a floor but names no domain_row and gives no floor_override_reason",
        )
    if domain_row:
        report.check()
        capability = matrix.get(domain_row)
        if capability is None:
            report.fail(
                where,
                f"domain_row {domain_row!r} is not a capability of the quantum_error_correction domain",
            )
        else:
            matrix_status = capability.get("status")
            if (status, matrix_status) in CONTRADICTIONS:
                report.fail(
                    where,
                    f"claims {status!r} while parity matrix row {domain_row!r} says {matrix_status!r}",
                )
            # Maturity belongs to capability-maturity.toml and the matrix row
            # already names the entry (or entries) it points at. A checklist row
            # that names an entry the matrix row does not is the drift this check
            # exists to stop: two documents describing the same capability would
            # otherwise be free to disagree about what it is. A divergence is
            # allowed only with a stated reason, because a matrix row can cover
            # more than one capability and then it can only point at some of them.
            matrix_maturity = matrix_maturity_refs(capability)
            if matrix_maturity and maturity_ref not in matrix_maturity:
                if not str(row.get("maturity_ref_override_reason", "")).strip():
                    report.fail(
                        where,
                        f"maturity_ref {maturity_ref!r} is not one the parity matrix row "
                        f"{domain_row!r} names ({', '.join(repr(r) for r in matrix_maturity)}), "
                        "and the row states no maturity_ref_override_reason",
                    )
            priority = capability.get("priority")
            if floor and priority and floor != priority:
                if not str(row.get("floor_override_reason", "")).strip():
                    report.fail(
                        where,
                        f"floor {floor!r} disagrees with parity matrix priority {priority!r}"
                        f" for domain_row {domain_row!r} and states no floor_override_reason",
                    )

    return {
        "id": row_id,
        "status": status,
        "matrix_row": domain_row,
        "matrix_status": matrix_status,
        "present": len(present),
        "absent": len(absent),
    }


def check_field(
    row: dict[str, Any],
    index: int,
    repo_root: Path,
    index_of: SymbolIndex,
    report: Report,
) -> dict[str, Any]:
    row_id = row.get("id") or f"field[{index}]"
    where = f"field {row_id}"
    report.check()

    verdict = row.get("verdict")
    if verdict not in FIELD_VERDICTS:
        report.fail(where, f"verdict {verdict!r} is not one of {FIELD_VERDICTS}")

    cudaq_symbol = str(row.get("cudaq_symbol", "")).strip()
    if not cudaq_symbol:
        report.fail(
            where, "cudaq_symbol is empty; state the CUDA-Q side or write '(none)'"
        )

    target = str(row.get("flagquantum_symbol", "")).strip()
    negative = row.get("negative_search", []) or []

    if verdict == "absent":
        # The CUDA-Q side has it and FlagQuantum has nothing to point at.
        if target:
            report.fail(where, "verdict `absent` must leave flagquantum_symbol empty")
        if not negative:
            report.fail(
                where, "verdict `absent` needs a negative_search proof of the absence"
            )
    elif verdict == "extra":
        # FlagQuantum has it and the CUDA-Q surface has no counterpart.
        if not target:
            report.fail(
                where,
                "verdict `extra` must name the flagquantum_symbol that has no counterpart",
            )
    else:
        if not target:
            report.fail(where, f"verdict {verdict!r} requires a flagquantum_symbol")

    if target:
        report.check()
        resolved = index_of.resolves(target)
        if verdict in ("equivalent", "renamed", "reshaped", "extra") and not resolved:
            report.fail(where, f"verdict {verdict!r} but {target!r} does not resolve")
        if verdict == "absent" and resolved:
            report.fail(where, f"verdict `absent` but {target!r} resolves")
        member = target.split(".")[-1]
        if member in index_of.all_names and not resolved:
            report.fail(where, f"{member!r} exists in the tree but not at {target!r}")

    for entry in negative:
        kind, _, proof = entry.partition(":")
        if kind == "symbol":
            report.check()
            if not proof:
                report.fail(where, f"negative_search {entry!r} names no symbol")
            elif index_of.resolves(proof):
                report.fail(
                    where,
                    f"negative_search {entry!r} now resolves -- the diff is stale",
                )
        else:
            report.check()
            if (repo_root / entry).exists():
                report.fail(
                    where, f"negative_search {entry!r} now exists -- the diff is stale"
                )

    if not str(row.get("note", "")).strip():
        report.fail(
            where, "note is empty; a diff row without a stated basis is an assertion"
        )

    # A row whose upstream side was not read from the pinned baseline has to say
    # so in the row, not only in the file header, because the row is what a
    # reader acts on.
    if row.get("provenance_unverified") and "UNVERIFIED" not in str(
        row.get("note", "")
    ):
        report.fail(
            where, "provenance_unverified is set but the note does not say UNVERIFIED"
        )

    return {"id": row_id, "verdict": verdict, "flagquantum_symbol": target}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo-root", default=DEFAULT_REPO_ROOT, help="FlagQuantum checkout to check"
    )
    parser.add_argument(
        "--checklist", default=DEFAULT_CHECKLIST, help="checklist TOML to check"
    )
    parser.add_argument(
        "--reading",
        default=DEFAULT_READING,
        help="companion prose document; every row id must be mentioned in it",
    )
    parser.add_argument("--json", action="store_true", help="emit a JSON report")
    parser.add_argument("--quiet", action="store_true", help="print only failures")
    args = parser.parse_args(argv)

    checklist_path = Path(args.checklist).resolve()
    repo_root = Path(args.repo_root).resolve()

    report = Report()
    if not checklist_path.is_file():
        print(f"checklist not found: {checklist_path}", file=sys.stderr)
        return 2
    if not repo_root.is_dir():
        print(f"repository root not found: {repo_root}", file=sys.stderr)
        return 2

    checklist = load_toml(checklist_path)
    index_of = SymbolIndex(repo_root / "flagquantum", "flagquantum")
    for failure in index_of.parse_failures:
        report.fail("symbol index", f"unparsable source: {failure}")

    matrix, domain_surface = qec_matrix_rows(repo_root / "contracts")
    maturity = maturity_entries(repo_root)
    provenance = contract_version_provenance(repo_root / "contracts")

    # The checklist names the tool that checks it. A pointer that no longer
    # resolves is how a reader ends up running nothing and believing the rows.
    report.check()
    declared_tool = str(checklist.get("checked_by", ""))
    if declared_tool and not (repo_root / declared_tool).is_file():
        report.fail(
            "header",
            f"checked_by names {declared_tool!r}, which is not a file in the repository",
        )

    # A checklist that admits any unverified row must state the limit of its own
    # provenance once, in the header, so the caveat cannot live only in one row.
    all_rows = list(checklist.get("rows", [])) + list(checklist.get("fields", []))
    if any(row.get("provenance_unverified") for row in all_rows):
        report.check()
        if not str(checklist.get("provenance_limit", "")).strip():
            report.fail(
                "header",
                "rows carry provenance_unverified but the checklist states no provenance_limit",
            )

    # The header's provenance is the contract's. A limit stated in two documents
    # is a limit that can drift in one of them, so the component version the
    # rows were read at is checked against the contract that records it -- and
    # its absence is a failure rather than a pass, because the checklist's whole
    # caveat rests on the two release lines being separate.
    report.check()
    if not provenance:
        report.fail(
            "header",
            "contracts/cudaq-parity-matrix.toml records no baseline.version_provenance, "
            "so the checklist's provenance_limit rests on nothing",
        )
    else:
        recorded = provenance.get("component_version_recorded")
        line = list(provenance.get("component_release_line", []))
        report.check()
        if checklist.get("baseline_component_version") != recorded:
            report.fail(
                "header",
                f"baseline_component_version {checklist.get('baseline_component_version')!r} "
                f"is not the component version the parity contract records ({recorded!r})",
            )
        report.check()
        if line and checklist.get("baseline_component_version") not in line:
            report.fail(
                "header",
                "baseline_component_version is not on the release line the parity "
                f"contract records ({', '.join(line)})",
            )
        report.check()
        source = str(checklist.get("baseline_component_source", ""))
        recorded_source = provenance.get("component_source")
        if (
            not source
            or not isinstance(recorded_source, str)
            or recorded_source not in source
        ):
            report.fail(
                "header",
                "baseline_component_source does not name the release index the parity "
                f"contract records ({recorded_source!r})",
            )

    rows = [
        check_row(row, i, repo_root, index_of, matrix, domain_surface, maturity, report)
        for i, row in enumerate(checklist.get("rows", []))
    ]
    fields = [
        check_field(row, i, repo_root, index_of, report)
        for i, row in enumerate(checklist.get("fields", []))
    ]

    # The reading document is where a person meets the checklist, so a row that
    # exists only in the TOML is invisible work. Checked only when the file is
    # there, so the checker still stands alone.
    reading_path = Path(args.reading).resolve()
    if reading_path.is_file():
        reading = reading_path.read_text(encoding="utf-8")
        for entry in rows + fields:
            report.check()
            if f"`{entry['id']}`" not in reading:
                report.fail(
                    f"reading {reading_path.name}",
                    f"row {entry['id']!r} is not mentioned, so it is invisible to a reader",
                )

    if args.json:
        print(
            json.dumps(
                {
                    "repo_root": str(repo_root),
                    "checklist": str(checklist_path),
                    "checks": report.checks,
                    "rows": rows,
                    "fields": fields,
                    "failures": report.failures,
                    "ok": report.ok(),
                },
                indent=2,
            )
        )
        return 0 if report.ok() else 1

    if not args.quiet:
        print(f"checklist : {checklist_path.name}")
        print(f"repository: {repo_root}")
        print(f"matrix    : {len(matrix)} quantum_error_correction capability rows")
        print(f"maturity  : {len(maturity)} registered capabilities")
        print()
        print(f"{'row':<34} {'status':<11} {'matrix row':<26} {'verdict':<8} in/out")
        print("-" * 96)
        for entry in rows:
            print(
                f"{entry['id']:<34} {entry['status']!s:<11} "
                f"{entry['matrix_row']!s:<26} {'':<8} "
                f"{entry['present']}/{entry['absent']}"
            )
        for entry in fields:
            print(
                f"{entry['id']:<34} {'':<11} {'':<26} {entry['verdict']!s:<8} "
                f"{entry['flagquantum_symbol'] or '-'}"
            )
        print()

    if report.ok():
        print(
            f"OK: {report.checks} checks passed, {len(rows)} alignment rows, {len(fields)} field rows"
        )
        return 0

    print(
        f"FAIL: {len(report.failures)} of {report.checks} checks failed",
        file=sys.stderr,
    )
    for failure in report.failures:
        print(f"  - {failure}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
