import pytest

from flagquantum.deployment import braket_provider, provider_utils, providers

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "name",
    (
        "AmazonBraketProvider",
        "BraketSubmissionPreview",
        "braket_backend_profile",
    ),
)
def test_provider_aggregator_preserves_braket_object_identity(name):
    assert getattr(providers, name) is getattr(braket_provider, name)


@pytest.mark.parametrize(
    "name",
    (
        "_normalize_counts",
        "_flip_counts",
        "_unwrap_result_envelope",
        "_extract_counts",
        "_strip_barrier",
        "_format_circuit_source",
    ),
)
def test_provider_aggregator_preserves_utility_function_identity(name):
    assert getattr(providers, name) is getattr(provider_utils, name)
