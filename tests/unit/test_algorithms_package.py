from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import flagquantum.algorithms as algorithms
from flagquantum.algorithms import Hamiltonian, run_vqe

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_canonical_algorithms_package_owns_public_implementations() -> None:
    assert Hamiltonian is algorithms.Hamiltonian
    assert run_vqe is algorithms.run_vqe
    assert algorithms.OptimizationStage.__module__ == (
        "flagquantum.algorithms.optimization"
    )
    assert algorithms.Hamiltonian.__module__ == "flagquantum.algorithms.core"


def test_algorithms_stack_has_been_removed() -> None:
    code = """
import importlib.util
assert importlib.util.find_spec("flagquantum.algorithms_stack") is None
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
