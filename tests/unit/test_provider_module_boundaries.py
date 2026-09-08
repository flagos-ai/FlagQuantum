import pytest

import flagquantum.deployment as deployment
from flagquantum.deployment import providers
from flagquantum.providers.execution import (
    braket,
    http,
    local,
    quafu,
    quafu_calibration,
    result_parsing,
)

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


def test_provider_aggregator_preserves_quafu_object_identity():
    assert providers.QuafuProvider is quafu.QuafuProvider


def test_deployment_preserves_quafu_calibration_object_identity():
    assert (
        deployment.quafu_noise_model_from_chip_info
        is quafu_calibration.quafu_noise_model_from_chip_info
    )


def test_deployment_preserves_local_provider_object_identity():
    assert deployment.LocalSimulatorProvider is local.LocalSimulatorProvider


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
