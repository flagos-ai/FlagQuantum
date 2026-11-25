from pathlib import Path

import pytest
import yaml

from tools.validate_sbom import sbom_errors

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_supply_chain_has_public_egress_and_remains_a_required_fail_closed_gate():
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    job = jobs["supply-chain"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 20
    assert job["permissions"] == {"contents": "read"}
    assert not job.get("continue-on-error", False)
    assert "supply-chain" in jobs["ci-complete"]["needs"]
    steps = job["steps"]
    audit = next(step for step in steps if "pip-audit --strict" in step.get("run", ""))
    assert audit["timeout-minutes"] == 8
    assert (
        "python tools/write_audit_requirements.py audit-requirements.txt"
        in audit["run"]
    )
    assert (
        "pip-audit --strict --disable-pip --no-deps "
        "--requirement audit-requirements.txt --timeout 60"
    ) in audit["run"]
    commands = [step.get("run", "") for step in steps]
    for command in (
        "bandit -q -lll -r flagquantum",
        "cyclonedx-py environment",
        "python tools/validate_sbom.py sbom.cdx.json",
    ):
        assert any(command in run for run in commands)
    for step in steps:
        assert not step.get("continue-on-error", False)
        assert "|| true" not in step.get("run", "")


def test_ci_has_coverage_security_and_sbom_gates():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    coverage_config = (ROOT / ".coveragerc").read_text()
    assert "python tools/check_coverage.py" in workflow
    assert "flagquantum/benchmarking/*" in coverage_config
    assert "flagquantum/kernels/triton/*" in coverage_config
    assert "python tools/write_audit_requirements.py audit-requirements.txt" in workflow
    assert "pip-audit --strict --disable-pip --no-deps" in workflow
    assert "bandit -q -lll -r flagquantum" in workflow
    assert "cyclonedx-py environment" in workflow
    assert "tools/validate_sbom.py" in workflow


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
