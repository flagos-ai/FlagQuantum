#!/usr/bin/env python3
"""Validate the qubit-vocabulary ledger against a live scan of the package.

The ledger in `contracts/qubit-vocabulary-contract.toml` is the only progress
record for the `wire` to `qubit` migration. This gate is what makes it a record
rather than a document:

- the baseline parameter list is frozen, and the live scan must equal the
  baseline minus the retired identifiers, so a slice can only shrink the set;
- a wire-named parameter that no ledger entry covers fails, including one added
  by a change that had nothing to do with vocabulary;
- a retired identifier that is still present fails, so a slice cannot claim
  progress it did not make;
- a deprecated alias must still forward to the qubit name the ledger records.

"Only shrink, never grow" needs the equality, not two inequalities: with a
subset check alone, deleting a baseline line would look like progress, and with
a superset check alone, a new site would look like the baseline already covered
it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from census_wire_vocabulary import (  # noqa: E402
    DECLARATION_KINDS,
    all_public_attribute_names,
    attribute_census,
    census,
    definition_census,
    public_parameter_names,
    supersedes,
)

CONTRACT = ROOT / "contracts" / "qubit-vocabulary-contract.toml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
PRE_PUSH = ROOT / "tools" / "pre_push.py"
SELF_PATH = "tools/check_qubit_vocabulary.py"

EXPECTED_HEADER = {
    "schema": "flagquantum_qubit_vocabulary_contract_v1",
    "maturity": "development_evidence",
    "census_tool": "tools/census_wire_vocabulary.py",
    "gate_tool": SELF_PATH,
    "ir_version": "1.0",
    "ir_version_unchanged": True,
}


def _load_toml(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], tomllib.loads(path.read_text(encoding="utf-8")))


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return sorted(repeated)


def contract_errors(
    contract: dict[str, Any], *, package_root: Path | None = None
) -> tuple[str, ...]:
    errors: list[str] = []

    for name, value in EXPECTED_HEADER.items():
        if contract.get(name) != value:
            errors.append(f"qubit vocabulary contract {name} must be {value!r}")

    naming = contract.get("naming", {})
    if naming.get("root_wire") != "qubit" or naming.get("root_wires") != "qubits":
        errors.append("qubit vocabulary naming roots drifted")
    if naming.get("variadic_prefix_preserved") is not True:
        errors.append("qubit vocabulary variadic names must keep their prefix")
    forbidden = tuple(naming.get("forbidden", ()))
    if not forbidden:
        errors.append("qubit vocabulary contract must name the forbidden alternatives")

    boundary = contract.get("boundary", {})
    if boundary.get("surface") != "public_function_surface":
        errors.append("qubit vocabulary boundary surface drifted")
    if boundary.get("private_excluded") is not True:
        errors.append("qubit vocabulary boundary must exclude private code")
    if not isinstance(boundary.get("definition"), str) or not boundary["definition"]:
        errors.append("qubit vocabulary boundary must state its definition")

    exclusions = contract.get("exclusions", {})
    for name in (
        "serialized_keys",
        "environment_variables",
        "serialized_reason",
        "environment_reason",
    ):
        if not exclusions.get(name):
            errors.append(f"qubit vocabulary exclusion {name} is missing")

    surfaces = contract.get("other_surfaces", {})
    for name in (
        "public_attribute_names",
        "public_definition_names",
        "message_strings",
    ):
        if int(surfaces.get(name, 0)) <= 0:
            errors.append(f"qubit vocabulary other surface {name} must be measured")
    for name in (
        "public_attribute_owner",
        "public_attribute_condition",
        "public_definition_owner",
        "public_definition_condition",
        "message_strings_owner",
        "message_strings_condition",
    ):
        if not isinstance(surfaces.get(name), str) or not surfaces[name]:
            errors.append(f"qubit vocabulary other surface {name} is unowned")

    ledger = [str(entry) for entry in contract.get("ledger", {}).get("canonical", ())]
    retirement = [
        str(entry) for entry in contract.get("retirement", {}).get("sites", ())
    ]
    alias_rows = list(contract.get("aliases", {}).get("declared", ()))
    alias_baseline = {str(row["site"]): str(row["replacement"]) for row in alias_rows}
    if not alias_rows or any(
        "site" not in row or "replacement" not in row for row in alias_rows
    ):
        errors.append("qubit vocabulary alias ledger must record site and replacement")
    removal_version = contract.get("aliases", {}).get("removal_version")
    if not isinstance(removal_version, str) or not removal_version:
        errors.append("qubit vocabulary alias ledger must name a removal version")

    for label, values in (("ledger", ledger), ("retirement", retirement)):
        repeated = _duplicates(values)
        if repeated:
            errors.append(f"qubit vocabulary {label} repeats {repeated[:3]}")
    repeated = _duplicates([*ledger, *alias_baseline])
    if repeated:
        errors.append(f"qubit vocabulary baseline repeats {repeated[:3]}")

    baseline = set(ledger) | set(alias_baseline)
    if len(ledger) != int(boundary.get("measured_canonical", -1)):
        errors.append(
            "qubit vocabulary ledger length must equal boundary.measured_canonical"
        )
    if len(alias_baseline) != int(boundary.get("measured_deprecated_aliases", -1)):
        errors.append(
            "qubit vocabulary alias ledger length must equal "
            "boundary.measured_deprecated_aliases"
        )
    unknown = sorted(set(retirement) - baseline)
    if unknown:
        errors.append(
            f"qubit vocabulary retirement names sites outside the baseline: {unknown[:3]}"
        )

    scanned = census(package_root)
    scanned_attributes = attribute_census(package_root)
    declarations = {
        site.identifier: site.declaration
        for site in (*scanned_attributes.ledgered, *scanned_attributes.excluded)
    }
    live_canonical = {site.identifier for site in scanned.canonical}
    live_aliases = {site.identifier: site.replacement for site in scanned.aliases}
    live = live_canonical | set(live_aliases)
    expected = baseline - set(retirement)

    appeared = sorted(live - expected)
    if appeared:
        errors.append(
            "qubit vocabulary found wire-named parameters outside the ledger: "
            f"{appeared[:3]}"
        )
    missing = sorted(expected - live)
    if missing:
        errors.append(
            "qubit vocabulary ledger entries are neither live nor retired: "
            f"{missing[:3]}"
        )
    for identifier, replacement in sorted(live_aliases.items()):
        declared = alias_baseline.get(identifier)
        if declared is None:
            continue
        if declared != replacement or not supersedes(
            identifier.rsplit("::", 1)[1], replacement
        ):
            errors.append(
                f"qubit vocabulary alias {identifier} must forward to {declared!r}"
            )

    if not scanned.internal:
        errors.append(
            "qubit vocabulary private bucket is empty; the boundary is vacuous"
        )
    if len(scanned.internal) != int(boundary.get("measured_private", -1)):
        errors.append(
            "qubit vocabulary private bucket changed size; "
            "re-measure boundary.measured_private"
        )

    declared_parameters = public_parameter_names(package_root)
    declared_attributes = {
        identifier.rsplit("::", 1)[1]
        for identifier in all_public_attribute_names(package_root)
    }
    for alternative in forbidden:
        if alternative in declared_parameters:
            errors.append(
                f"qubit vocabulary forbids the alternative spelling {alternative!r}"
            )
        if alternative in declared_attributes:
            errors.append(
                f"qubit vocabulary forbids the alternative spelling {alternative!r} "
                "as an attribute name"
            )

    errors.extend(_attribute_errors(contract, scanned_attributes))
    errors.extend(_definition_errors(contract, package_root))

    slices = list(contract.get("slices", ()))
    if not slices:
        errors.append("qubit vocabulary contract must declare its migration slices")
    identifiers: list[str] = []
    owned: dict[str, str] = {}
    attribute_ledger = [
        str(entry)
        for entry in contract.get("attribute_ledger", {}).get("canonical", ())
    ]
    attribute_rows = list(contract.get("attribute_exclusions", {}).get("sites", ()))
    definition_rows = list(contract.get("definition_ledger", {}).get("sites", ()))
    retired_kinds = _retired_attribute_kinds(contract)
    attribute_paths = {
        identifier.split("::", 1)[0] for identifier in attribute_ledger
    } | {str(row.get("site", "")).split("::", 1)[0] for row in attribute_rows}
    attribute_paths |= {
        str(row.get("site", "")).split("::", 1)[0] for row in definition_rows
    }
    attribute_paths.discard("")
    baseline_paths = {
        identifier.split("::", 1)[0] for identifier in baseline
    } | attribute_paths
    for entry in slices:
        slice_id = str(entry.get("id", ""))
        if not slice_id:
            errors.append("qubit vocabulary slice is missing an id")
            continue
        identifiers.append(slice_id)
        files = [str(path) for path in entry.get("files", ())]
        if not files:
            errors.append(f"qubit vocabulary slice {slice_id} owns no files")
        for path in files:
            owner = owned.get(path)
            if owner is not None:
                errors.append(
                    f"qubit vocabulary path {path} is owned by {owner} and {slice_id}"
                )
            owned[path] = slice_id
            if not (ROOT / path).is_file():
                errors.append(f"qubit vocabulary slice {slice_id} names missing {path}")
        for name in ("title", "owner_team"):
            if not entry.get(name):
                errors.append(f"qubit vocabulary slice {slice_id} is missing {name}")
        owned_files = set(files)
        expected_count = sum(
            1 for identifier in ledger if identifier.split("::", 1)[0] in owned_files
        )
        if int(entry.get("canonical_count", -1)) != expected_count:
            errors.append(
                f"qubit vocabulary slice {slice_id} canonical_count must be "
                f"{expected_count}"
            )
        expected_names = sum(
            1
            for identifier in attribute_ledger
            if identifier.split("::", 1)[0] in owned_files
        )
        if int(entry.get("attribute_count", -1)) != expected_names:
            errors.append(
                f"qubit vocabulary slice {slice_id} attribute_count must be "
                f"{expected_names}"
            )
        for label in DECLARATION_KINDS:
            expected_declared = sum(
                1
                for identifier in attribute_ledger
                if identifier.split("::", 1)[0] in owned_files
                and (declarations.get(identifier) or retired_kinds.get(identifier))
                == label
            )
            if int(entry.get(f"attribute_{label}_count", -1)) != expected_declared:
                errors.append(
                    f"qubit vocabulary slice {slice_id} attribute_{label}_count must "
                    f"be {expected_declared}"
                )
        expected_persisted = sum(
            1
            for row in attribute_rows
            if str(row.get("site", "")).split("::", 1)[0] in owned_files
        )
        if int(entry.get("persisted_attribute_count", -1)) != expected_persisted:
            errors.append(
                f"qubit vocabulary slice {slice_id} persisted_attribute_count must be "
                f"{expected_persisted}"
            )
        expected_definitions = sum(
            1
            for row in definition_rows
            if str(row.get("site", "")).split("::", 1)[0] in owned_files
        )
        if int(entry.get("definition_count", -1)) != expected_definitions:
            errors.append(
                f"qubit vocabulary slice {slice_id} definition_count must be "
                f"{expected_definitions}"
            )
    repeated = _duplicates(identifiers)
    if repeated:
        errors.append(f"qubit vocabulary slice ids repeat {repeated[:3]}")
    unowned = sorted(baseline_paths - set(owned))
    if unowned:
        errors.append(f"qubit vocabulary baseline paths are unowned: {unowned[:3]}")
    stray = sorted(set(owned) - baseline_paths)
    if stray:
        errors.append(
            f"qubit vocabulary slices own paths with no baseline site: {stray[:3]}"
        )

    for name, raw_path in contract.get("verification", {}).items():
        if not isinstance(raw_path, str) or not (ROOT / raw_path).is_file():
            errors.append(f"qubit vocabulary verification path {name!r} does not exist")
    for name in ("proposal", "authorization", "naming_reference", "predecessor"):
        raw_path = contract.get(name)
        if not isinstance(raw_path, str) or not (ROOT / raw_path).is_file():
            errors.append(f"qubit vocabulary reference {name!r} does not exist")

    for gated in (CI_WORKFLOW, PRE_PUSH):
        if SELF_PATH not in gated.read_text(encoding="utf-8"):
            errors.append(f"qubit vocabulary gate is not wired into {gated.name}")
    return tuple(errors)


def _attribute_errors(
    contract: dict[str, Any],
    scanned: Any,
) -> list[str]:
    """Reconcile the attribute surface, and report the files it implicates.

    The parameter ledger and the attribute ledger answer the same question about
    two different surfaces, so the same shape of check applies: the baseline is
    frozen, a slice may only move identifiers from the baseline to the
    retirement list, and the live scan must equal baseline minus retirement. The
    part with no parameter counterpart is the split. An attribute is either a
    name a slice will rename or a spelling that reaches a payload, the census
    decides which, and a disagreement in either direction has to fail -- that is
    what stops a rename from silently changing a serialized key.
    """

    errors: list[str] = []
    ledger = [
        str(entry)
        for entry in contract.get("attribute_ledger", {}).get("canonical", ())
    ]
    rows = list(contract.get("attribute_exclusions", {}).get("sites", ()))
    retirement_rows = [
        entry
        for entry in contract.get("attribute_retirement", {}).get("sites", ())
        if isinstance(entry, dict)
    ]
    retirement = [str(row.get("site", "")) for row in retirement_rows]
    for row in retirement_rows:
        for name in ("site", "replacement", "declaration"):
            if not isinstance(row.get(name), str) or not row[name]:
                errors.append(
                    f"qubit vocabulary attribute retirement is missing {name}: {row}"
                )
        if row.get("declaration") not in DECLARATION_KINDS:
            errors.append(
                "qubit vocabulary attribute retirement must name the declaration "
                f"kind it discharged: {row}"
            )
        name = str(row.get("site", "")).rsplit("::", 1)[-1]
        if not supersedes(name, str(row.get("replacement", ""))):
            errors.append(
                f"qubit vocabulary attribute retirement {row.get('site')} must be "
                f"replaced by {row.get('replacement')!r}"
            )

    if not ledger:
        errors.append("qubit vocabulary contract must ledger the attribute surface")
    if not rows:
        errors.append(
            "qubit vocabulary attribute exclusions must be named sites, not a pattern"
        )
    for row in rows:
        for name in (
            "site",
            "replacement",
            "declaration",
            "witness",
            "evidence",
            "reason",
            "removal_condition",
        ):
            if not isinstance(row.get(name), str) or not row[name]:
                errors.append(
                    f"qubit vocabulary attribute exclusion is missing {name}: {row}"
                )

    baseline = set(ledger) | {str(row.get("site", "")) for row in rows}
    baseline.discard("")
    for label, values in (
        ("attribute ledger", ledger),
        ("attribute retirement", retirement),
    ):
        repeated = _duplicates(values)
        if repeated:
            errors.append(f"qubit vocabulary {label} repeats {repeated[:3]}")
    unknown = sorted(set(retirement) - baseline)
    if unknown:
        errors.append(
            "qubit vocabulary attribute retirement names sites outside the "
            f"baseline: {unknown[:3]}"
        )

    measured_ledger = int(
        contract.get("attribute_ledger", {}).get("measured_canonical", -1)
    )
    measured_persisted = int(
        contract.get("attribute_exclusions", {}).get("measured_persisted", -1)
    )
    if len(ledger) != measured_ledger:
        errors.append(
            "qubit vocabulary attribute ledger length must equal "
            "attribute_ledger.measured_canonical"
        )
    if len(rows) != measured_persisted:
        errors.append(
            "qubit vocabulary exclusion count must equal "
            "attribute_exclusions.measured_persisted"
        )
    surfaces = contract.get("other_surfaces", {})
    measured_total = int(surfaces.get("public_attribute_names", -1))
    if measured_ledger + measured_persisted != measured_total:
        errors.append(
            "qubit vocabulary attribute split must account for every name: "
            f"{measured_ledger} + {measured_persisted} != {measured_total}"
        )
    for name, measured in (
        ("public_attribute_ledgered", measured_ledger),
        ("public_attribute_persisted", measured_persisted),
    ):
        if int(surfaces.get(name, -1)) != measured:
            errors.append(
                f"qubit vocabulary other surface {name} must equal the ledger it "
                "describes"
            )

    live_names = {site.identifier for site in scanned.ledgered}
    live_persisted = {site.identifier: site for site in scanned.excluded}
    by_site = {str(row.get("site", "")): row for row in rows}
    retired = set(retirement)

    appeared = sorted(live_names - (set(ledger) - retired))
    if appeared:
        errors.append(
            "qubit vocabulary found wire-named attributes outside the ledger: "
            f"{appeared[:3]}"
        )
    missing = sorted((set(ledger) - retired) - live_names)
    if missing:
        errors.append(
            "qubit vocabulary attribute ledger entries are neither live nor "
            f"retired: {missing[:3]}"
        )

    for identifier, site in sorted(live_persisted.items()):
        row = by_site.get(identifier)
        if row is None:
            if identifier not in retired:
                errors.append(
                    "qubit vocabulary attribute reaches a payload but is not "
                    f"excluded by name: {identifier} (witness {site.witness})"
                )
            continue
        if (
            row.get("witness") != site.witness
            or row.get("evidence") != site.evidence
            or row.get("declaration") != site.declaration
        ):
            errors.append(
                f"qubit vocabulary exclusion {identifier} no longer matches the "
                f"scan: witness {site.witness!r}, evidence {site.evidence!r}, "
                f"declaration {site.declaration!r}"
            )
        declared = str(row.get("replacement", ""))
        if declared != site.replacement or not supersedes(site.attribute, declared):
            errors.append(
                f"qubit vocabulary attribute {identifier} must be replaced by "
                f"{site.replacement!r}"
            )

    for row in rows:
        identifier = str(row.get("site", ""))
        if identifier in retired:
            if identifier in live_persisted:
                errors.append(
                    f"qubit vocabulary retired attribute {identifier} still "
                    "reaches a payload"
                )
            continue
        if identifier not in live_persisted:
            errors.append(
                f"qubit vocabulary exclusion {identifier} is stale: its class no "
                "longer builds a payload from that name"
            )

    return errors


def _retired_attribute_kinds(contract: dict[str, Any]) -> dict[str, str]:
    """The declaration kind each retired attribute site carried when it was frozen.

    The per-slice counts split the work by declaration kind, and a retired site is
    no longer in the live scan. Reading the kind from the retirement row keeps the
    slice table stable across slices, so `12 + 9 + 7` stays the record of what WQ-2
    owed instead of silently collapsing to zero rows of nothing.
    """

    return {
        str(row.get("site", "")): str(row.get("declaration", ""))
        for row in contract.get("attribute_retirement", {}).get("sites", ())
        if isinstance(row, dict)
    }


def _definition_errors(
    contract: dict[str, Any],
    package_root: Path | None,
) -> list[str]:
    """Reconcile the module-level definition names against a live scan.

    A renamed parameter inside a function whose own name still says `wire` is not
    a finished migration, and neither of the other two ledgers sees the name. The
    check is the same shape as the other two: frozen baseline, retirement list,
    live scan equals baseline minus retirement, and every retired name must
    actually be gone.
    """

    errors: list[str] = []
    rows = list(contract.get("definition_ledger", {}).get("sites", ()))
    retirement = [
        str(entry)
        for entry in contract.get("definition_retirement", {}).get("sites", ())
    ]
    if not rows:
        errors.append("qubit vocabulary contract must ledger its definition names")
    for row in rows:
        for name in ("site", "replacement", "kind"):
            if not isinstance(row.get(name), str) or not row[name]:
                errors.append(
                    f"qubit vocabulary definition ledger is missing {name}: {row}"
                )
    baseline = {str(row.get("site", "")) for row in rows}
    baseline.discard("")
    measured = int(contract.get("definition_ledger", {}).get("measured_canonical", -1))
    if len(baseline) != measured:
        errors.append(
            "qubit vocabulary definition ledger length must equal "
            "definition_ledger.measured_canonical"
        )
    surfaces = contract.get("other_surfaces", {})
    if int(surfaces.get("public_definition_names", -1)) != measured:
        errors.append(
            "qubit vocabulary other surface public_definition_names must equal the "
            "ledger it describes"
        )
    repeated = _duplicates([str(row.get("site", "")) for row in rows])
    if repeated:
        errors.append(f"qubit vocabulary definition ledger repeats {repeated[:3]}")
    unknown = sorted(set(retirement) - baseline)
    if unknown:
        errors.append(
            "qubit vocabulary definition retirement names sites outside the "
            f"baseline: {unknown[:3]}"
        )

    scanned = definition_census(package_root)
    live = {site.identifier: site for site in scanned.ledgered}
    retired = set(retirement)
    appeared = sorted(set(live) - (baseline - retired))
    if appeared:
        errors.append(
            "qubit vocabulary found wire-named public definitions outside the "
            f"ledger: {appeared[:3]}"
        )
    missing = sorted((baseline - retired) - set(live))
    if missing:
        errors.append(
            "qubit vocabulary definition ledger entries are neither live nor "
            f"retired: {missing[:3]}"
        )
    still_live = sorted(retired & set(live))
    if still_live:
        errors.append(
            "qubit vocabulary retired public definitions still exist: "
            f"{still_live[:3]}"
        )
    for row in rows:
        identifier = str(row.get("site", ""))
        site = live.get(identifier)
        if site is None:
            continue
        declared = str(row.get("replacement", ""))
        if declared != site.replacement or not supersedes(site.name, declared):
            errors.append(
                f"qubit vocabulary definition {identifier} must be replaced by "
                f"{site.replacement!r}"
            )
        if str(row.get("kind", "")) != site.kind:
            errors.append(
                f"qubit vocabulary definition {identifier} kind changed to "
                f"{site.kind!r}"
            )
    return errors


def slice_progress(
    contract: dict[str, Any], surface: str = "parameter"
) -> tuple[tuple[str, int, int], ...]:
    """Retired and remaining baseline sites per slice, for the gate's report."""

    surface_sections = {
        "parameter": ("retirement", "ledger", "canonical"),
        "attribute": ("attribute_retirement", "attribute_ledger", "canonical"),
        "definition": ("definition_retirement", "definition_ledger", "sites"),
    }
    retirement_section, ledger_section, key = surface_sections[surface]
    retired = set()
    for entry in contract.get(retirement_section, {}).get("sites", ()):
        retired.add(
            str(entry.get("site", "")) if isinstance(entry, dict) else str(entry)
        )
    ledger: list[str] = []
    for entry in contract.get(ledger_section, {}).get(key, ()):
        if isinstance(entry, dict):
            ledger.append(str(entry.get("site", "")))
        else:
            ledger.append(str(entry))
    rows: list[tuple[str, int, int]] = []
    for entry in contract.get("slices", ()):
        files = set(str(path) for path in entry.get("files", ()))
        sites = [item for item in ledger if item.split("::", 1)[0] in files]
        done = sum(1 for item in sites if item in retired)
        rows.append((str(entry["id"]), done, len(sites) - done))
    return tuple(rows)


def _report(contract: dict[str, Any]) -> None:
    for surface, label in (
        ("parameter", "baseline sites"),
        ("attribute", "attribute sites"),
        ("definition", "definition names"),
    ):
        rows = slice_progress(contract, surface)
        retired = sum(done for _, done, _ in rows)
        remaining = sum(left for _, _, left in rows)
        print(
            f"Qubit vocabulary contract passed: {retired} of {retired + remaining} "
            f"{label} retired"
        )
        for slice_id, done, left in rows:
            print(f"  {slice_id:5s} retired {done:3d}  remaining {left:3d}")


def main() -> int:
    contract = _load_toml(CONTRACT)
    errors = contract_errors(contract)
    if errors:
        print("\n".join(errors))
        return 1
    _report(contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
