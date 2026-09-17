"""The coverage gate must enforce the floor it advertises."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.check_coverage import (
    EXPECTED_SCHEMA,
    load_toml,
    parse_coverage,
    policy_errors,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "contracts" / "coverage-policy.toml"


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


def test_parse_coverage_reads_the_measured_source_root(tmp_path: Path):
    xml = tmp_path / "coverage.xml"
    xml.write_text(
        '<?xml version="1.0" ?><coverage line-rate="0.7">'
        '<packages><package name="runtime" line-rate="0.5"/></packages>'
        "</coverage>",
        encoding="utf-8",
    )

    global_rate, packages = parse_coverage(xml)

    assert global_rate == pytest.approx(70.0)
    assert packages == {"runtime": pytest.approx(50.0)}
