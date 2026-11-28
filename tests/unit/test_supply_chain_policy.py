from pathlib import Path

import pytest

from tools.architecture.check_dependency_policy import load_toml
from tools.release.validate_sbom import sbom_errors

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_ci_has_coverage_security_and_sbom_gates():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    coverage_config = load_toml(ROOT / "pyproject.toml")["tool"]["coverage"]["run"][
        "omit"
    ]
    assert "python tools/ci/check_coverage.py" in workflow
    assert "flagquantum/benchmarking/*" in coverage_config
    assert "flagquantum/simulation/kernels/triton/*" in coverage_config
    assert (
        "python tools/validation/write_audit_requirements.py audit-requirements.txt"
        in workflow
    )
    assert "pip-audit --strict --disable-pip --no-deps" in workflow
    assert "bandit -q -lll -r flagquantum" in workflow
    assert "cyclonedx-py environment" in workflow
    assert "tools/release/validate_sbom.py" in workflow


def test_sbom_validator_fails_closed_and_accepts_minimum_valid_payload():
    assert sbom_errors({}) != ()
    assert (
        sbom_errors(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.6",
                "components": [{"name": "flagquantum", "version": "0.1.0"}],
            }
        )
        == ()
    )


def test_supply_chain_audits_a_fresh_dependency_environment():
    import yaml

    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    job = workflow["jobs"]["supply-chain"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["concurrency"] == {"group": "flagquantum-gh-2", "queue": "max"}
    assert job["timeout-minutes"] == 20
    steps = job["steps"]
    isolation = [
        step
        for step in steps
        if step.get("uses") == "./.github/actions/isolated-python"
    ]
    assert len(isolation) == 1
    assert isolation[0]["with"]["inherit-installed-dependencies"] == "false"
    audit = next(
        step for step in steps if step.get("name") == "Audit installed dependencies"
    )
    assert "pip-audit --strict --disable-pip --no-deps" in audit["run"]
    assert "--ignore-vuln" not in audit["run"]
    assert audit["timeout-minutes"] == 8
    assert "--timeout 60" in audit["run"]
