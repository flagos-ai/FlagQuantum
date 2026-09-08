import pytest

import flagquantum.deployment as deployment
import flagquantum.remote as remote
from flagquantum.remote.qpu import braket, calibration, http, quafu
from flagquantum.testing import InMemoryRemoteTarget

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("name", "module"),
    (
        ("AmazonBraketProvider", braket),
        ("BraketSubmissionPreview", braket),
        ("braket_backend_profile", braket),
        ("HttpQuantumProvider", http),
        ("ProviderCredentials", http),
        ("ProviderEndpoints", http),
        ("QuantumCloudTransport", http),
        ("UrllibTransport", http),
        ("QuafuProvider", quafu),
        ("quafu_noise_model_from_chip_info", calibration),
    ),
)
def test_remote_facade_exports_maintained_adapters(name, module):
    assert getattr(remote, name) is getattr(module, name)


def test_deployment_owns_packages_not_concrete_remote_adapters():
    assert hasattr(deployment, "create_deployment_package")
    assert not hasattr(deployment, "QuafuProvider")
    assert not hasattr(deployment, "HttpQuantumProvider")


def test_local_control_plane_double_lives_in_testing():
    provider = InMemoryRemoteTarget()
    assert provider.provider == "local"
