"""Categorized public imports for remote compute adapters."""

import pytest

import flagquantum.remote.compute as remote_compute
from flagquantum.remote.compute import JiudingClient, JiudingCredentials
from flagquantum.remote.compute.jiuding import JiudingClient as JiudingImplementation
from flagquantum.remote.compute.jiuding import (
    JiudingCredentials as JiudingCredentialsImplementation,
)

pytestmark = pytest.mark.unit


def test_compute_entrypoint_exposes_jiuding_api() -> None:
    assert JiudingClient is JiudingImplementation
    assert JiudingCredentials is JiudingCredentialsImplementation
    assert remote_compute.__all__ == ("JiudingClient", "JiudingCredentials")
