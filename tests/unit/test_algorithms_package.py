from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import flagquantum.algorithms as algorithms
from flagquantum.algorithms import Hamiltonian, fold_program, run_vqe

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_canonical_algorithms_package_owns_public_implementations() -> None:
    assert Hamiltonian is algorithms.Hamiltonian
    assert run_vqe is algorithms.run_vqe
    assert fold_program is algorithms.fold_program
    assert algorithms.OptimizationStage.__module__ == (
        "flagquantum.algorithms.optimization"
    )
    assert algorithms.Hamiltonian.__module__ == "flagquantum.algorithms.core"
    assert algorithms.FoldingPlan.__module__ == "flagquantum.algorithms.folding"
    # The constrained optimizer lives in its own module rather than in the unit it
    # is measured against, so a caller reads which method they asked for from the
    # module the name resolves to.
    assert algorithms.CobylaOptimizer.__module__ == "flagquantum.algorithms.cobyla"
    assert algorithms.CobylaResult.__module__ == "flagquantum.algorithms.cobyla"
    # The folding unit is reachable as a name and as its own module, so a caller
    # never has to reach a private path to find it.
    assert algorithms.folding.fold_program is fold_program


def test_algorithms_stack_has_been_removed() -> None:
    code = """
import importlib.util
assert importlib.util.find_spec("flagquantum.algorithms_stack") is None
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
