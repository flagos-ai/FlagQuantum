"""Categorized public imports for remote compute adapters."""

import pytest

import flagquantum.compute.remote.compute as remote_compute
from flagquantum.compute.remote.compute import JiudingClient, JiudingCredentials
from flagquantum.compute.remote.compute.jiuding import (
    JiudingClient as JiudingImplementation,
)
from flagquantum.compute.remote.compute.jiuding import (
    JiudingCredentials as JiudingCredentialsImplementation,
)

pytestmark = pytest.mark.unit


def test_compute_entrypoint_exposes_jiuding_api() -> None:
    assert JiudingClient is JiudingImplementation
    assert JiudingCredentials is JiudingCredentialsImplementation
    assert remote_compute.__all__ == ("JiudingClient", "JiudingCredentials")
