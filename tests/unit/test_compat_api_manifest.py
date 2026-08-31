import pytest

import flagquantum as fq
import flagquantum.api as compatibility_api
import flagquantum.backends as fqb
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
    ],
)
def test_stable_manifest_split_preserves_export_identity(name):
    assert getattr(compatibility_api, name) is getattr(
        __import__("flagquantum", fromlist=[name]), name
    )


@pytest.mark.parametrize("name", ["run_mps", "run_tensor_network"])
def test_migrated_backend_exports_preserve_canonical_identity(name):
    assert getattr(compatibility_api, name) is getattr(fqb, name)
    assert name not in fq.__all__


@pytest.mark.parametrize("name", ["AmazonBraketProvider", "DistributedQuantumDevice"])
def test_historical_compatibility_exports_are_not_stable_root_contracts(name):
    assert hasattr(compatibility_api, name)
    assert name not in fq.__all__
    assert name not in dir(fq)
