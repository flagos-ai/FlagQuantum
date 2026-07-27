from pathlib import Path

import pytest

from tools.validate_sbom import sbom_errors

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_ci_has_coverage_security_and_sbom_gates():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "--cov-fail-under=50" in workflow
    assert "pip list --format freeze --exclude flagquantum" in workflow
    assert (
        "pip-audit --strict --no-deps --requirement audit-requirements.txt" in workflow
    )
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
