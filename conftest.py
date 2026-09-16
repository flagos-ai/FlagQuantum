# flagquantum/conftest.py
import os

# Must be set before any imports
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Imports follow env setup: torch reads threading env vars at import time.
from collections.abc import Iterator  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from flagquantum.testing.reset_caches import reset_caches  # noqa: E402

if os.environ.get("PYTEST_DEBUG_CONFIG"):
    print("=" * 60)
    print("Pytest configuration loaded")
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    print(f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}")
    print("=" * 60)


@pytest.fixture(autouse=True)
def _reset_quantum_caches() -> Iterator[None]:
    """Reset module-level caches before and after each test for isolation."""

    reset_caches()
    yield
    reset_caches()


@pytest.fixture(autouse=True)
def _seed_torch() -> None:
    """Seed the global torch RNG so tests are reproducible by default."""

    torch.manual_seed(0)
