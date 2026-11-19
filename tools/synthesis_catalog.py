"""Verify and render the classified, source-pinned synthesis example catalog."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import TypedDict, cast

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "tests/fixtures/classiq_qmod/corpus_catalog.json"


class Source(TypedDict):
    path: str
    git_blob_sha1: str
    sha256: str


class ApprovedSource(TypedDict):
    path: str
    sha256: str
    revised: bool
    change: str | None


class FileDigest(TypedDict):
    path: str
    sha256: str


class Case(TypedDict):
    id: str
    category: str
    topic: str
    directory: str
    original: Source
    approved: ApprovedSource
    source_url: str
    companion_files: list[FileDigest]
    classical_inputs: dict[str, float | list[float]] | None
    basis_mps: bool
    native_example: str | None
    native_comparison: str | None
    reference_comparison: str | None
    validation_sources: list[str]
    circuit_validation_sources: list[str]
    fixture_sources: list[str]
    recorded_original: dict[str, object]
    recorded_approved: dict[str, object]


class Catalog(TypedDict):
    schema: str
    upstream_commit: str
    implementation_commit: str
    license: FileDigest
    cases: list[Case]


def load_catalog() -> Catalog:
    return cast(Catalog, json.loads(CATALOG.read_text(encoding="utf-8")))


def example_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to((ROOT / "examples/synthesis").resolve()):
        raise ValueError(f"catalog path is outside the synthesis examples: {relative}")
    return path


def asset_path(relative: str) -> Path:
    """Resolve corpus evidence without exposing source files as user examples."""
    path = (ROOT / relative).resolve()
    roots = (ROOT / "examples/synthesis", ROOT / "tests/fixtures/classiq_qmod")
    if not any(path.is_relative_to(root.resolve()) for root in roots):
        raise ValueError(f"catalog asset is outside its owned roots: {relative}")
    return path


def verify_catalog(catalog: Catalog) -> None:
    cases = catalog["cases"]
    if (
        catalog["schema"] != "flagquantum.synthesis_examples.v1"
        or len(cases) != 216
        or len({case["id"] for case in cases}) != 216
        or len({case["directory"].casefold() for case in cases}) != 216
        or sum(case["approved"]["revised"] for case in cases) != 6
    ):
        raise ValueError(
            "catalog inventory or revision count differs from the pinned 216"
        )
    overlay = ROOT / "tests/fixtures/classiq_qmod/accepted_revision"
    review = json.loads((overlay / "manifest.json").read_text())
    revisions = {row["path"]: row for row in review["files"]}
    bindings = json.loads((overlay / "bindings.json").read_text())
    basis = set(json.loads((overlay / "basis-mps-paths.json").read_text()))
    if (
        catalog["upstream_commit"] != review["pinned_classiq_commit"]
        or {case["id"] for case in cases if case["approved"]["revised"]}
        != revisions.keys()
    ):
        raise ValueError("catalog source version or revision set differs from approval")
    if Counter(case["recorded_original"]["stage"] for case in cases) != {
        "execution_smoke_passed": 210,
        "frontend_refused": 6,
    }:
        raise ValueError("catalog original results differ from the pinned source audit")
    assets = [catalog["license"]]
    for case in cases:
        raw = asset_path(case["original"]["path"]).read_bytes()
        blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if blob != case["original"]["git_blob_sha1"]:
            raise ValueError(f"original source blob mismatch: {case['id']}")
        accepted = asset_path(case["approved"]["path"]).read_bytes()
        accepted_blob = hashlib.sha1(
            f"blob {len(accepted)}\0".encode() + accepted
        ).hexdigest()
        if (
            case["recorded_original"]["path"] != case["id"]
            or case["recorded_approved"]["path"] != case["id"]
            or case["recorded_original"]["blob"] != blob
            or case["recorded_approved"]["blob"] != accepted_blob
            or case["classical_inputs"] != bindings.get(case["id"])
            or case["basis_mps"] != (case["id"] in basis)
        ):
            raise ValueError(
                f"case identity, bindings or execution profile changed: {case['id']}"
            )
        if case["approved"]["revised"]:
            revision = revisions[case["id"]]
            if (
                case["original"]["sha256"] != revision["original_sha256"]
                or case["approved"]["sha256"] != revision["candidate_sha256"]
            ):
                raise ValueError(
                    f"case differs from its approved source revision: {case['id']}"
                )
        if case["approved"]["revised"] != (raw != accepted):
            raise ValueError(f"incorrect revision label: {case['id']}")
        if (
            case["category"]
            not in {"tutorials", "algorithms", "applications", "community", "functions"}
            or case["category"] != Path(case["id"]).parts[0]
        ):
            raise ValueError(f"unknown example category: {case['id']}")
        if case["recorded_approved"]["stage"] != "execution_smoke_passed":
            raise ValueError(f"missing recorded acceptance: {case['id']}")
        if bool(case["native_example"]) != bool(case["native_comparison"]):
            raise ValueError(f"incomplete native comparison: {case['id']}")
        if case["native_example"]:
            native = example_path(case["native_example"])
            provenance = json.loads(native.with_name("provenance.json").read_text())
            reference = case["reference_comparison"]
            if (
                provenance["upstream_path"] != case["id"]
                or provenance["original_sha256"] != case["original"]["sha256"]
                or reference is None
                or asset_path(reference).with_name("source.qmod").read_bytes() != raw
            ):
                raise ValueError(
                    f"native example is not tied to this original: {case['id']}"
                )
            native.read_bytes()
            comparison = example_path(cast(str, case["native_comparison"]))
            tree = ast.parse(comparison.read_text(), filename=str(comparison))
            if not any(
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
                and len(node.test.ops) == 1
                and isinstance(node.test.ops[0], ast.Eq)
                and len(node.test.comparators) == 1
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value == "__main__"
                and any(
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Call)
                    for statement in node.body
                )
                for node in tree.body
            ):
                raise ValueError(
                    f"missing executable comparison entry point: {case['id']}"
                )
        for reference in (
            *case["validation_sources"],
            *case["circuit_validation_sources"],
        ):
            if (
                not reference.startswith(("tests/", "tools/"))
                or not (ROOT / reference).is_file()
            ):
                raise ValueError(f"missing validation source: {reference}")
        for companion in case["companion_files"]:
            if companion["path"].endswith(".synthesis_options.json") and asset_path(
                companion["path"]
            ) not in {
                asset_path(source["path"]).with_suffix(".synthesis_options.json")
                for source in (case["original"], case["approved"])
            }:
                raise ValueError(
                    f"companion preferences are not adjacent to their source: {case['id']}"
                )
        assets.extend([case["original"], case["approved"], *case["companion_files"]])
    for asset in assets:
        if (
            hashlib.sha256(asset_path(asset["path"]).read_bytes()).hexdigest()
            != asset["sha256"]
        ):
            raise ValueError(f"catalog asset digest mismatch: {asset['path']}")


def _link(path: str, parent: Path, label: str) -> str:
    return f"[{label}]({os.path.relpath(ROOT / path, parent)})"


def case_readme(case: Case, catalog: Catalog, *, notebook_validated: bool) -> str:
    """Document the runnable native implementation, without an importer launcher."""
    parent = example_path(case["directory"])
    assert case["native_example"] and case["native_comparison"]
    main_module = ".".join(Path(case["native_example"]).with_suffix("").parts)
    comparison = ".".join(Path(case["native_comparison"]).with_suffix("").parts)
    comparison_scope = (
        [
            "The Circuit comparison checks the complete density matrix, register order",
            "and reset/hidden environment. Separate unitary-block tests check complex",
            "amplitudes and absolute phase. Neither comparison calls a source importer.",
        ]
        if case["recorded_original"].get("execution_profile") == "exact_density_matrix"
        else [
            "The Circuit comparison checks complete complex amplitudes, absolute phase,",
            "declared register order and hidden workspace. It does not call a source importer.",
        ]
    )
    return "\n".join(
        [
            "# " + Path(case["directory"]).name.replace("_", " "),
            "",
            "This example uses native FlagQuantum Python quantum functions.",
            "",
            "```sh",
            "python -m " + main_module,
            "python -m " + comparison,
            "```",
            "",
            *comparison_scope,
            "",
            *(
                [
                    "The quantum program and complete local Notebook workflow are validated.",
                    "The workflow scope and replacements for external services",
                    (
                        "are documented in [the workflow guide](WORKFLOW.md)."
                        if (parent / "WORKFLOW.md").is_file()
                        else "are recorded in `provenance.json`."
                    ),
                    "",
                ]
                if notebook_validated
                else [
                    "The validated scope is the quantum program. A related notebook may also include",
                    "training, data preparation or external execution that requires separate migration.",
                    *(
                        ["See the [quantum-program guide](WORKFLOW.md)."]
                        if (parent / "WORKFLOW.md").is_file()
                        else []
                    ),
                    "",
                ]
            ),
            f"Reference identity: `{case['id']}` at `{catalog['upstream_commit']}`.",
            "",
            "Provenance is recorded in `provenance.json`; original sources and importer",
            "regressions live only in test fixtures. The shared third-party license applies.",
            "",
            _link(catalog["license"]["path"], parent, "Third-party license"),
            "",
        ]
    )


def render(catalog: Catalog) -> dict[Path, str]:
    base = ROOT / "examples/synthesis"
    categories = ("algorithms", "applications", "community", "functions", "tutorials")
    available = [case for case in catalog["cases"] if case["native_example"]]
    inventory = json.loads(
        (ROOT / "tests/fixtures/classiq_qmod/native_library_inventory.json").read_text()
    )
    groups = inventory["groups"]
    native_paths = {str(path.relative_to(ROOT)) for path in base.rglob("*.py")}
    for group in groups:
        native = group["native_program"]
        if native and native["path"] not in native_paths:
            raise ValueError(
                f"missing or case-mismatched native program: {native['path']}"
            )
    source_count = sum(len(group["sources"]) for group in groups)
    notebooks = [group for group in groups if group["kind"] == "notebook_workflow"]
    validated_workflows = sum(
        group["workflow_status"] == "validated" for group in notebooks
    )
    validated_programs = {
        (group["id"], group["native_program"]["path"])
        for group in notebooks
        if group["workflow_status"] == "validated" and group["native_program"]
    }
    lines = [
        "# Native FlagQuantum example library",
        "",
        "User examples contain native Python programs and independent Circuit comparisons.",
        "Original source files and importer launchers are test/development evidence, not examples.",
        "",
        f"Recorded native quantum-program counterparts: **{len(available)}/216**.",
        "Counts reflect per-batch evidence, not a fresh all-program acceptance or release gate.",
        f"Validated local Notebook workflows: **{validated_workflows}/{len(notebooks)}**.",
        f"Expanded inventory: **{source_count}** source artifacts in **{len(groups)}** provisional path groups.",
        "Python groups still require main-case/helper review. A quantum-program counterpart",
        "does not certify its notebook workflow. Pending entries have no runnable placeholder.",
        "",
        "See [migration status](MIGRATION_STATUS.md) for the complete five-category inventory.",
        "",
    ]
    output = {CATALOG: json.dumps(catalog, separators=(",", ":")) + "\n"}
    for category in categories:
        entries = [case for case in available if case["category"] == category]
        lines += [f"## {category.capitalize()} ({len(entries)} native programs)", ""]
        category_lines = [
            "# " + category.capitalize(),
            "",
            "Native FlagQuantum programs currently available:",
            "",
        ]
        for case in entries:
            link = _link(
                case["directory"] + "/README.md", base, Path(case["directory"]).name
            )
            lines.append("- " + link)
            category_lines.append(
                "- "
                + _link(
                    case["directory"] + "/README.md",
                    base / category,
                    Path(case["directory"]).name,
                )
            )
            output[example_path(case["directory"]) / "README.md"] = case_readme(
                case,
                catalog,
                notebook_validated=(case["id"][:-5], case["native_example"])
                in validated_programs,
            )
        if not entries:
            category_lines += [
                "No upstream counterpart has completed native validation in this category yet."
            ]
        workflows = [
            group
            for group in notebooks
            if group["category"] == category and group["workflow_status"] == "validated"
        ]
        if workflows:
            category_lines += ["", "## Validated local Notebook workflows", ""]
            lines += ["", "Validated local Notebook workflows:", ""]
            for group in workflows:
                path = Path(group["native_program"]["path"]).with_name("README.md")
                category_lines.append(
                    "- " + _link(str(path), base / category, path.parent.name)
                )
                lines.append("- " + _link(str(path), base, path.parent.name))
            lines.append("")
        tracked = {case["native_example"] for case in available}
        tracked.update(group["native_program"]["path"] for group in workflows)
        additional = sorted(
            path
            for path in (base / category).rglob("main.py")
            if str(path.relative_to(ROOT)) not in tracked
        )
        if additional:
            category_lines += ["", "## Additional FlagQuantum teaching examples", ""]
            for path in additional:
                category_lines.append(
                    "- "
                    + _link(
                        str(path.relative_to(ROOT)), base / category, path.parent.name
                    )
                )
        category_lines += [
            "",
            "See the [full migration inventory](../MIGRATION_STATUS.md).",
            "",
        ]
        output[base / category / "README.md"] = "\n".join(category_lines)
        lines.append("")
    output[base / "CATALOG.md"] = "\n".join(lines).rstrip() + "\n"
    migration = [
        "# Five-category native migration",
        "",
        f"Pinned upstream commit: `{inventory['upstream_commit']}`.",
        "",
        "Groups use exact upstream paths without suffixes, so paired notebooks and",
        "quantum sources share one row. Python-only groups remain unclassified until",
        "main cases, helpers, data dependencies and verification criteria are reviewed.",
        "",
        "No notebook workflow is certified by a quantum-only counterpart.",
        f"Validated local Notebook workflows: **{validated_workflows}/{len(notebooks)}**.",
        "",
        "| Upstream group | Kind | Native quantum program | Full workflow |",
        "| --- | --- | --- | --- |",
    ]
    for group in groups:
        native = group["native_program"]
        link = _link(native["path"], base, "Native program") if native else "Pending"
        migration.append(
            f"| `{group['id']}` | {group['kind']} | {link} | {group['workflow_status']} |"
        )
    output[base / "MIGRATION_STATUS.md"] = "\n".join(migration) + "\n"
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="regenerate native navigation"
    )
    args = parser.parse_args()
    catalog = load_catalog()
    verify_catalog(catalog)
    for path, expected in render(catalog).items():
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected, encoding="utf-8")
        elif not path.is_file() or path.read_text(encoding="utf-8") != expected:
            raise SystemExit(
                f"catalog documentation is stale: {path.relative_to(ROOT)}"
            )
    print(
        "Verified pinned test sources, native provenance and five-category navigation"
    )


if __name__ == "__main__":
    main()
