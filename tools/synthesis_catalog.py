"""Verify and render the classified, source-pinned synthesis example catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import TypedDict, cast

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "examples/synthesis/catalog.json"


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
        raw = example_path(case["original"]["path"]).read_bytes()
        blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if blob != case["original"]["git_blob_sha1"]:
            raise ValueError(f"original source blob mismatch: {case['id']}")
        accepted = example_path(case["approved"]["path"]).read_bytes()
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
        if case["category"] not in {"tutorials", "algorithms", "applications"}:
            raise ValueError(f"unknown example category: {case['id']}")
        if case["recorded_approved"]["stage"] != "execution_smoke_passed":
            raise ValueError(f"missing recorded acceptance: {case['id']}")
        if bool(case["native_example"]) != bool(case["native_comparison"]):
            raise ValueError(f"incomplete native comparison: {case['id']}")
        if case["native_example"]:
            comparison = example_path(cast(str, case["native_comparison"]))
            if comparison.with_name("source.qmod").read_bytes() != raw:
                raise ValueError(
                    f"native example is not tied to this original: {case['id']}"
                )
            example_path(case["native_example"]).read_bytes()
        for reference in (
            *case["validation_sources"],
            *case["circuit_validation_sources"],
        ):
            if (
                not reference.startswith(("tests/", "tools/"))
                or not (ROOT / reference).is_file()
            ):
                raise ValueError(f"missing validation source: {reference}")
        assets.extend([case["original"], case["approved"], *case["companion_files"]])
    for asset in assets:
        if (
            hashlib.sha256(example_path(asset["path"]).read_bytes()).hexdigest()
            != asset["sha256"]
        ):
            raise ValueError(f"catalog asset digest mismatch: {asset['path']}")


def _link(path: str, parent: Path, label: str) -> str:
    return f"[{label}]({os.path.relpath(ROOT / path, parent)})"


def case_readme(case: Case, catalog: Catalog) -> str:
    parent = example_path(case["directory"])
    original = case["original"]
    accepted = case["approved"]
    costs = case["recorded_approved"]
    lines = [
        f"# {Path(case['id']).stem.replace('_', ' ')}",
        "",
        f"Category: {case['category']} / {case['topic']}.",
        "",
        f"[Original Classiq source]({case['source_url']}) at `{catalog['upstream_commit']}`.",
        "",
        "## Implementations",
        "",
        "| Form | Availability |",
        "| --- | --- |",
        f"| Qmod original | {_link(original['path'], parent, 'source.qmod')} |",
        "| Native Python qfunc | "
        + (
            _link(case["native_example"], parent, "Corresponding native program")
            if case["native_example"]
            else "Not yet provided for this exact original"
        )
        + " |",
        "| Circuit comparison | "
        + (
            _link(case["native_comparison"], parent, "Runnable three-form comparison")
            if case["native_comparison"]
            else "See the validation source links below"
        )
        + " |",
        "",
        "## Run",
        "",
        "Run from the repository root with FlagQuantum installed.",
        "",
        "```sh",
        f"python -m examples.synthesis.run_qmod '{case['id']}'"
        + (" --revision approved" if accepted["revised"] else ""),
        "```",
        "",
        "The shared runner binds the recorded classical inputs and selects the",
        "recorded exact CPU execution profile. It prints measured resources and",
        "execution validation, using the same audit implementation as corpus acceptance.",
        "",
        "Classical bindings used by this entry:",
        "",
        "```json",
        json.dumps(case["classical_inputs"], indent=2),
        "```",
        "",
        "## Verification",
        "",
        f"Recorded implementation commit: `{catalog['implementation_commit']}`.",
        "",
        f"Original result: `{case['recorded_original']['stage']}`. Approved result: `{costs['stage']}`.",
        "",
        f"Recorded resources: {costs['qubits']} physical qubits, {costs['operations']} operations, depth {costs['depth']}.",
        "",
        "Batch execution checks successful exact execution and basic result validity.",
        "Linked tests and validators document their own input bindings, full-state",
        "references, global-phase and workspace checks; inspect their individual scopes.",
        "",
    ]
    if case["native_example"] and case["native_comparison"]:
        lines += [
            "Run the native program and its three-form comparison:",
            "",
            "```sh",
            "python -m " + ".".join(Path(case["native_example"]).with_suffix("").parts),
            "python -m "
            + ".".join(Path(case["native_comparison"]).with_suffix("").parts),
            "```",
            "",
        ]
    if accepted["revised"]:
        lines += [
            "## Approved numerical revision",
            "",
            str(accepted["change"]),
            "",
            _link(accepted["path"], parent, "Approved source"),
            "",
            "The original remains byte-identical and is expected to be refused.",
            "Run the command without `--revision approved` to inspect that refusal.",
            "",
            "Recorded original diagnostic:",
            "",
            "```text",
            str(case["recorded_original"]["detail"]),
            "```",
            "",
        ]
    lines += ["## Existing validation sources", ""]
    if case["validation_sources"]:
        lines += [
            "- " + _link(path, parent, Path(path).name)
            for path in case["validation_sources"]
        ]
    else:
        lines += [
            "A case-specific validation source has not yet been linked in this catalog."
        ]
    lines += [
        "",
        "Circuit-based validation source links: "
        + str(len(case["circuit_validation_sources"]))
        + ".",
        "",
        "## Provenance",
        "",
        f"Original Git blob: `{original['git_blob_sha1']}`.",
        "",
        f"Original SHA-256: `{original['sha256']}`.",
        "",
        f"Approved SHA-256: `{accepted['sha256']}`.",
        "",
        _link(catalog["license"]["path"], parent, "Classiq MIT license"),
        "",
    ]
    return "\n".join(lines)


def render(catalog: Catalog) -> dict[Path, str]:
    base = ROOT / "examples/synthesis"
    counts = Counter(case["category"] for case in catalog["cases"])
    native_count = sum(bool(case["native_example"]) for case in catalog["cases"])
    reference_count = sum(
        bool(case["circuit_validation_sources"]) for case in catalog["cases"]
    )
    lines = [
        "# Classified Classiq Qmod cases",
        "",
        f"All **216** pinned original paths are classified here. **{native_count}** have a byte-matched corresponding native example; the other native entries are explicitly marked missing.",
        "",
        f"**{reference_count}** cases link existing Circuit validation sources. A link locates a validation implementation; its own bindings and scope determine what it proves.",
        "",
        "The approved numerical revision passes 216/216 execution smoke checks. The original snapshot has 210 passes and six source-semantic refusals.",
        "",
        "```sh",
        "python -m examples.synthesis.run_qmod --verify",
        "python -m examples.synthesis.run_qmod --all --revision approved --output /tmp/qmod-examples.json",
        "```",
        "",
        "Use each case README for exact bindings, resource records, sources and validation links. Teaching examples in the other indexes may use smaller illustrative programs.",
        "",
    ]
    for category in ("tutorials", "algorithms", "applications"):
        lines += [f"## {category.capitalize()} ({counts[category]})", ""]
        topics = sorted(
            {case["topic"] for case in catalog["cases"] if case["category"] == category}
        )
        for topic in topics:
            lines += [
                f"### {topic.replace('_', ' ')}",
                "",
                "| Original case | Native qfunc | Circuit validation links | Original status |",
                "| --- | --- | --- | --- |",
            ]
            for case in catalog["cases"]:
                if (case["category"], case["topic"]) != (category, topic):
                    continue
                link = _link(case["directory"] + "/README.md", base, case["id"])
                lines.append(
                    f"| {link} | {'Available' if case['native_example'] else 'Not provided'} | {len(case['circuit_validation_sources'])} | {case['recorded_original']['stage']} |"
                )
            lines.append("")
    output = {base / "CATALOG.md": "\n".join(lines)}
    output.update(
        {
            example_path(case["directory"]) / "README.md": case_readme(case, catalog)
            for case in catalog["cases"]
        }
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="regenerate case READMEs and the classified index",
    )
    args = parser.parse_args()
    catalog = load_catalog()
    verify_catalog(catalog)
    for path, expected in render(catalog).items():
        if args.write:
            path.write_text(expected, encoding="utf-8")
        elif not path.is_file() or path.read_text(encoding="utf-8") != expected:
            raise SystemExit(
                f"catalog documentation is stale: {path.relative_to(ROOT)}"
            )
    print(
        "Verified 216 original sources, six approved revisions, license, links and classified documentation"
    )


if __name__ == "__main__":
    main()
