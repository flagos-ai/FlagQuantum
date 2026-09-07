import pytest

from flagquantum.deployment import providers
from flagquantum.providers.execution import braket, http, result_parsing

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
    assert getattr(providers, name) is getattr(braket, name)


@pytest.mark.parametrize(
    "name",
    (
        "HttpQuantumProvider",
        "ProviderCredentials",
        "ProviderEndpoints",
        "QuantumCloudTransport",
        "UrllibTransport",
    ),
)
def test_provider_aggregator_preserves_http_object_identity(name):
    assert getattr(providers, name) is getattr(http, name)


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
    assert getattr(providers, name) is getattr(result_parsing, name)
