from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from tools.check_capability_maturity import (
    EXPECTED_LEVELS,
    REQUIRED_USER_FIELDS,
    aggregate_claim_value,
    claim_values,
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


def test_internal_experiments_may_have_no_public_api() -> None:
    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    capability = data["capabilities"]["split_real_imag_statevector_p5_autograd_bridge"]
    assert capability["level"] == "experimental"
    assert capability["public_apis"] == []
    assert maturity_errors(data, ROOT) == ()


def test_supported_capability_requires_a_public_api() -> None:
    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    data["capabilities"]["local_statevector"]["public_apis"] = []
    assert any(
        "local_statevector: supported or certified capability requires a public API"
        in error
        for error in maturity_errors(data, ROOT)
    )


def test_user_discovery_links_fail_closed_when_missing():
    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    data["capabilities"]["ir"]["quick_start"] = "examples/does_not_exist.py"
    errors = maturity_errors(data, ROOT)
    assert "ir: quick_start path does not exist: examples/does_not_exist.py" in errors


def test_performance_claim_binds_artifact_hash_code_environment_and_maturity():
    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    claim = data["capabilities"]["sharded_mps_training"]["performance_claims"][0]
    assert claim["artifact_sha256"]
    assert claim["code_version"]
    assert claim["environment"]
    assert claim["maturity"] == "development_evidence"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("artifact_sha256", "0" * 64, "artifact_sha256 mismatch"),
        ("code_version", "unknown", "code_version does not match artifact commit"),
        ("maturity", "experimental", "claim maturity must equal capability level"),
    ),
)
def test_performance_claim_identity_fails_closed(field, value, message):
    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    claim = data["capabilities"]["sharded_mps_training"]["performance_claims"][0]
    claim[field] = value
    assert any(message in error for error in maturity_errors(data, ROOT))


def test_performance_claim_fails_closed_when_artifact_or_environment_is_missing():
    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    claim = data["capabilities"]["sharded_mps_training"]["performance_claims"][0]
    claim["artifact"] = "benchmarks/results/local/does-not-exist.json"
    errors = maturity_errors(data, ROOT)
    assert any("artifact does not exist" in error for error in errors)

    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    claim = data["capabilities"]["sharded_mps_training"]["performance_claims"][0]
    claim["artifact"] = "README.md"
    errors = maturity_errors(data, ROOT)
    assert any("artifact must be under benchmarks/results" in error for error in errors)

    data = tomllib.loads((ROOT / "capability-maturity.toml").read_text())
    claim = data["capabilities"]["sharded_mps_training"]["performance_claims"][0]
    claim["environment"] = []
    assert any("missing environment" in error for error in maturity_errors(data, ROOT))


def test_claim_selector_aggregates_artifact_values():
    values = claim_values({"ranks": [{"value": 2}, {"value": 3}]}, "ranks[].value")
    assert aggregate_claim_value(values, "max") == 3
