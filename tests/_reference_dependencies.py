"""Keep optional core collection distinct from mandatory reference execution."""

import os
from typing import NoReturn

import pytest


def unavailable_reference(error: ModuleNotFoundError) -> NoReturn:
    """Retain the original error in required runs; explicitly skip optional runs."""
    if os.environ.get("FLAGQUANTUM_REQUIRE_FTQC_DEPENDENCIES", "0") != "0":
        raise error
    pytest.skip(
        f"Optional reference dependency is unavailable: {error}",
        allow_module_level=True,
    )
