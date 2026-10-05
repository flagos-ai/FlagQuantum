#!/usr/bin/env python3
"""Validate the qubit-vocabulary ledger against a live scan of the package.

The ledger in `contracts/qubit-vocabulary-contract.toml` is the only progress
record for the `wire` to `qubit` migration. This gate is what makes it a record
rather than a document:

- the ledger must equal what the live scan finds, minus the retired identifiers,
  so the set can only shrink and it shrinks in one of two recorded ways: a
  renamed parameter stays in the ledger and moves to the retirement list, while a
  site whose module no longer exists leaves the ledger and takes the declared
  baseline down with it;
- a wire-named parameter that no ledger entry covers fails, including one added
  by a change that had nothing to do with vocabulary;
- a retired identifier that is still present fails, so a slice cannot claim
  progress it did not make;
- a deprecated alias must still forward to the qubit name the ledger records.

"Only shrink, never grow" needs the equality, not two inequalities: with a
subset check alone, deleting a baseline line would look like progress, and with
a superset check alone, a new site would look like the baseline already covered
it. The equality is what makes a deletion from the ledger safe: it is only
accepted when the site is gone from the live scan too, so the row cannot be
dropped to silence a rename that has not happened.
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
    census,
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
    for name in ("public_attribute_names", "message_strings"):
        if int(surfaces.get(name, 0)) <= 0:
            errors.append(f"qubit vocabulary other surface {name} must be measured")
    for name in (
        "public_attribute_owner",
        "public_attribute_condition",
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
    for alternative in forbidden:
        if alternative in declared_parameters:
            errors.append(
                f"qubit vocabulary forbids the alternative spelling {alternative!r}"
            )

    slices = list(contract.get("slices", ()))
    if not slices:
        errors.append("qubit vocabulary contract must declare its migration slices")
    identifiers: list[str] = []
    owned: dict[str, str] = {}
    baseline_paths = {identifier.split("::", 1)[0] for identifier in baseline}
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
        expected_count = sum(
            1 for identifier in ledger if identifier.split("::", 1)[0] in set(files)
        )
        if int(entry.get("canonical_count", -1)) != expected_count:
            errors.append(
                f"qubit vocabulary slice {slice_id} canonical_count must be "
                f"{expected_count}"
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


def slice_progress(contract: dict[str, Any]) -> tuple[tuple[str, int, int], ...]:
    """Retired and remaining baseline sites per slice, for the gate's report."""

    retired = set(contract.get("retirement", {}).get("sites", ()))
    ledger = [str(entry) for entry in contract.get("ledger", {}).get("canonical", ())]
    rows: list[tuple[str, int, int]] = []
    for entry in contract.get("slices", ()):
        files = set(str(path) for path in entry.get("files", ()))
        sites = [item for item in ledger if item.split("::", 1)[0] in files]
        done = sum(1 for item in sites if item in retired)
        rows.append((str(entry["id"]), done, len(sites) - done))
    return tuple(rows)


def main() -> int:
    contract = _load_toml(CONTRACT)
    errors = contract_errors(contract)
    if errors:
        print("\n".join(errors))
        return 1
    rows = slice_progress(contract)
    retired = sum(done for _, done, _ in rows)
    remaining = sum(left for _, _, left in rows)
    print(
        f"Qubit vocabulary contract passed: {retired} of {retired + remaining} "
        "baseline sites retired"
    )
    for slice_id, done, left in rows:
        print(f"  {slice_id:5s} retired {done:3d}  remaining {left:3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
