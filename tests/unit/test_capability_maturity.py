from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from tools.check_capability_maturity import (
    EXPECTED_LEVELS,
    REQUIRED_USER_FIELDS,
    maturity_errors,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_repository_capability_maturity_matrix_is_valid():
    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    assert maturity_errors(data, ROOT) == ()


def test_levels_are_explicit_and_not_collapsed():
    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    assert tuple(data["levels"]) == EXPECTED_LEVELS
    assert (
        data["capabilities"]["sharded_mps_training"]["level"] == "development_evidence"
    )
    assert data["capabilities"]["tensor_network_training"]["level"] == "experimental"
    assert data["capabilities"]["local_statevector"]["level"] == "production_supported"
    assert data["capabilities"]["ir"]["level"] == "release_certified"


def test_release_certification_fails_closed_without_release_evidence():
    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    data["capabilities"]["local_statevector"]["level"] = "release_certified"
    errors = maturity_errors(data, ROOT)
    assert "local_statevector: release_certified requires release_gate" in errors
    assert "local_statevector: release_certified requires release_artifact" in errors


def test_every_capability_has_user_discovery_metadata():
    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    for capability in data["capabilities"].values():
        assert set(REQUIRED_USER_FIELDS) <= set(capability)


def test_user_discovery_links_fail_closed_when_missing():
    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    data["capabilities"]["ir"]["quick_start"] = "examples/does_not_exist.py"
    errors = maturity_errors(data, ROOT)
    assert "ir: quick_start path does not exist: examples/does_not_exist.py" in errors
