"""The coverage gate must enforce the floor it advertises."""

from __future__ import annotations

import configparser
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tools.ci.check_coverage import (
    EXPECTED_SCHEMA,
    load_toml,
    parse_coverage,
    policy_errors,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "contracts" / "coverage-policy.toml"
COVERAGERC = ROOT / ".coveragerc"


def test_global_floor_rejects_coverage_below_the_floor():
    policy = {"schema": EXPECTED_SCHEMA, "global": {"min": 60}}

    assert policy_errors(policy, 60.0, {}) == []
    assert "below floor 60%" in policy_errors(policy, 59.9, {})[0]


def test_package_floor_is_enforced_for_the_named_package():
    policy = {
        "schema": EXPECTED_SCHEMA,
        "global": {"min": 60},
        "packages": {"runtime": {"min": 55}, "twin": {"min": 50}},
    }
    packages = {"runtime": 55.0, "twin": 52.0}

    assert policy_errors(policy, 61.0, packages) == []
    errors = policy_errors(policy, 61.0, {**packages, "runtime": 54.9})
    assert errors == ["package runtime coverage 54.9% is below floor 55%"]


def test_unknown_package_name_fails_closed():
    policy = {
        "schema": EXPECTED_SCHEMA,
        "global": {"min": 60},
        "packages": {"flagquantum.runtime": {"min": 55}},
    }

    errors = policy_errors(policy, 61.0, {"runtime": 90.0})

    assert errors == ["package flagquantum.runtime coverage 0.0% is below floor 55%"]


def test_unenforceable_policy_shapes_are_reported():
    assert policy_errors({"schema": "wrong"}, 61.0, {}) == [
        f"coverage policy schema must be {EXPECTED_SCHEMA}",
        "coverage policy is missing [global]",
    ]
    assert policy_errors(
        {"schema": EXPECTED_SCHEMA, "global": {"min": "high"}}, 61.0, {}
    ) == ["[global].min must be a number"]
    assert policy_errors(
        {
            "schema": EXPECTED_SCHEMA,
            "global": {"min": 60},
            "packages": {"twin": 50},
        },
        61.0,
        {"twin": 90.0},
    ) == ["[packages.twin].min must be a number"]


def test_shipped_policy_keys_match_coverage_xml_package_names():
    """A misspelled key would silently clear its own floor."""
    policy = load_toml(POLICY)

    assert policy["schema"] == EXPECTED_SCHEMA
    assert isinstance(policy["global"]["min"], (int, float))
    for package, cfg in (policy.get("packages") or {}).items():
        assert isinstance(cfg, dict) and isinstance(
            cfg.get("min"), (int, float)
        ), f"[packages.{package}] must declare a numeric min"
        assert not package.startswith("flagquantum."), (
            f"[packages.{package}] must use the coverage XML package name, "
            "which is relative to the measured source root"
        )


def _measurement(
    root: Path,
    files: dict[str, tuple[str, dict[int, int]]],
    moves: list[dict] | None = None,
) -> Path:
    """Write source and actual line hits; package summaries are deliberately absent."""
    document = ET.Element("coverage")
    packages = ET.SubElement(document, "packages")
    covered = total = 0
    for filename, (text, hits) in files.items():
        path = root / "flagquantum" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        package = ET.SubElement(packages, "package", name=str(Path(filename).parent))
        classes = ET.SubElement(package, "classes")
        entry = ET.SubElement(classes, "class", filename=filename)
        lines = ET.SubElement(entry, "lines")
        for number, count in hits.items():
            ET.SubElement(lines, "line", number=str(number), hits=str(count))
            covered += int(count > 0)
            total += 1
    document.set("line-rate", str(covered / total))
    xml = root / "coverage.xml"
    ET.ElementTree(document).write(xml, encoding="unicode")
    if moves is not None:
        ledger = root / "docs/development/migration/independent-structure-moves.json"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(json.dumps({"moves": moves}), encoding="utf-8")
    return xml


def test_parse_coverage_reads_the_measured_source_root(tmp_path: Path):
    xml = _measurement(
        tmp_path,
        {"runtime/example.py": ("first = 1\nsecond = 2\n", {1: 1, 2: 0})},
    )
    global_rate, packages = parse_coverage(xml, tmp_path)
    assert global_rate == pytest.approx(50.0)
    assert packages == {"runtime": pytest.approx(50.0)}


def test_moved_code_keeps_its_original_floor_and_counts_public_facade_lines(tmp_path):
    xml = _measurement(
        tmp_path,
        {
            "new/engine.py": ("def compute():\n    return 2\n", {1: 1, 2: 1}),
            "old/engine.py": ("from flagquantum.new.engine import compute\n", {1: 0}),
        },
        [
            {
                "old": "flagquantum/old/engine.py",
                "new": "flagquantum/new/engine.py",
                "symbols": ["compute"],
            }
        ],
    )
    global_rate, packages = parse_coverage(xml, tmp_path)
    assert global_rate == pytest.approx(200 / 3)
    assert packages == {"old": pytest.approx(200 / 3)}
    policy = {
        "schema": EXPECTED_SCHEMA,
        "global": {"min": 60},
        "packages": {"old": {"min": 80}},
    }
    assert "below floor 80%" in policy_errors(policy, global_rate, packages)[0]


def test_partial_move_cannot_hide_missing_hits_in_a_well_covered_destination(tmp_path):
    xml = _measurement(
        tmp_path,
        {
            "destination/merged.py": (
                "def moved():\n    return 1\ndef existing():\n    return 2\n",
                {1: 1, 2: 0, 3: 1, 4: 1},
            )
        },
        [
            {
                "old": "flagquantum/restricted/old.py",
                "new": "flagquantum/destination/merged.py",
                "symbols": ["moved"],
                "partial": True,
            }
        ],
    )
    global_rate, packages = parse_coverage(xml, tmp_path)
    assert global_rate == 75.0
    assert packages == {"restricted": 50.0, "destination": 100.0}
    policy = {
        "schema": EXPECTED_SCHEMA,
        "global": {"min": 60},
        "packages": {"restricted": {"min": 90}},
    }
    assert policy_errors(policy, global_rate, packages) == [
        "package restricted coverage 50.0% is below floor 90%"
    ]


def test_merge_preserves_each_source_scope_without_double_counting_globals(tmp_path):
    xml = _measurement(
        tmp_path,
        {
            "destination/merged.py": (
                "shared = 1\ndef first():\n    return 1\ndef second():\n    return 2\n",
                {1: 1, 2: 1, 3: 0, 4: 1, 5: 1},
            )
        },
        [
            {
                "old": "flagquantum/alpha/a.py",
                "new": "flagquantum/destination/merged.py",
                "symbols": ["first"],
                "partial": True,
            },
            {
                "old": "flagquantum/beta/b.py",
                "new": "flagquantum/destination/merged.py",
                "symbols": ["second"],
                "partial": True,
            },
        ],
    )
    global_rate, packages = parse_coverage(xml, tmp_path)
    assert global_rate == 80.0
    assert packages == {"destination": 100.0, "alpha": 50.0, "beta": 100.0}


def test_a_missing_declared_definition_does_not_donate_destination_coverage(tmp_path):
    xml = _measurement(
        tmp_path,
        {"destination/new.py": ("def unrelated():\n    return 1\n", {1: 1, 2: 1})},
        [
            {
                "old": "flagquantum/restricted/old.py",
                "new": "flagquantum/destination/new.py",
                "symbols": ["missing"],
            }
        ],
    )
    global_rate, packages = parse_coverage(xml, tmp_path)
    assert packages == {"destination": 100.0}
    policy = {
        "schema": EXPECTED_SCHEMA,
        "global": {"min": 60},
        "packages": {"restricted": {"min": 90}},
    }
    assert policy_errors(policy, global_rate, packages) == [
        "package restricted coverage 0.0% is below floor 90%"
    ]


def test_package_summaries_without_measured_lines_are_rejected(tmp_path):
    xml = tmp_path / "coverage.xml"
    xml.write_text(
        '<coverage line-rate="1"><packages><package name="restricted" line-rate="1"/></packages></coverage>'
    )
    with pytest.raises(ValueError, match="no measured source lines"):
        parse_coverage(xml, tmp_path)


def test_a_measured_line_cannot_be_counted_twice(tmp_path):
    xml = _measurement(tmp_path, {"runtime/code.py": ("value = 1\n", {1: 1})})
    document = ET.parse(xml)
    lines = document.find("./packages/package/classes/class/lines")
    assert lines is not None
    ET.SubElement(lines, "line", number="1", hits="1")
    document.write(xml)
    with pytest.raises(ValueError, match="duplicate measured line"):
        parse_coverage(xml, tmp_path)


def test_raw_coverage_data_is_portable_across_shard_runners() -> None:
    config = configparser.ConfigParser()
    config.read(COVERAGERC, encoding="utf-8")

    assert config.get("run", "source") == "flagquantum"
    assert config.getboolean("run", "relative_files") is True
