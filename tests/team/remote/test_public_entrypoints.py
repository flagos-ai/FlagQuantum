"""Categorized public imports for remote compute adapters."""

import flagquantum.remote.compute as remote_compute
from flagquantum.remote.compute import JiudingClient
from flagquantum.remote.compute.jiuding import JiudingClient as JiudingImplementation


def test_compute_entrypoint_exposes_jiuding_client() -> None:
    assert JiudingClient is JiudingImplementation
    assert remote_compute.__all__ == ("JiudingClient",)
