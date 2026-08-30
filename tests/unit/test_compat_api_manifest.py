import pytest

import flagquantum.api as compatibility_api
from flagquantum import _compat_api_exports

pytestmark = pytest.mark.unit


def test_compatibility_manifest_matches_api_surface():
    expected = _compat_api_exports.compatibility_exports(
        devices=compatibility_api.devices,
        drawer=compatibility_api.drawer,
        encoding=compatibility_api.encoding,
        measurement=compatibility_api.measurement,
        ops=compatibility_api.ops,
        utils=compatibility_api.utils,
    )

    assert compatibility_api.__all__ == expected
    assert all(hasattr(compatibility_api, name) for name in expected)


@pytest.mark.parametrize(
    "name",
    [
        "Circuit",
        "ExecutionResult",
        "Module",
        "run",
        "run_mps",
        "run_tensor_network",
        "AmazonBraketProvider",
        "DistributedQuantumDevice",
    ],
)
def test_manifest_split_preserves_export_identity(name):
    assert getattr(compatibility_api, name) is getattr(
        __import__("flagquantum", fromlist=[name]), name
    )
