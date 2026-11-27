# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 guided_hamiltonian.py at
# bcb17bb25360ed68ce3bc084645db3e929799389.

"""Build the two guided examples through native k-XOR dependency owners."""

from sympy import symbols

from flagquantum.algorithms.kikuchi_block_encoding import KikuchiHamiltonian
from flagquantum.algorithms.kikuchi_guiding_state import GuidingState
from flagquantum.algorithms.kxor_instance import KXorInstance
from flagquantum.algorithms.primitives.black_box_prepare import PackedPrepareDeclaration
from flagquantum.algorithms.primitives.guided_hamiltonian_resource_model import (
    GuidedHamiltonianModel,
    GuidedPhaseEstimationModel,
)
from flagquantum.core.value_math import SymbolicInt, ceil, log2


def _problem(
    n: SymbolicInt,
    k: SymbolicInt,
    c: SymbolicInt,
    guide_constraints: SymbolicInt,
    solve_constraints: SymbolicInt,
) -> tuple[KikuchiHamiltonian, PackedPrepareDeclaration]:
    ell = c * k
    guide = GuidingState(KXorInstance.symbolic(n, guide_constraints, k, max_rhs=2), ell)
    block = KikuchiHamiltonian(
        KXorInstance.symbolic(n, solve_constraints, k, max_rhs=2),
        ell,
        ell * ceil(log2(n)),
    )
    return block, PackedPrepareDeclaration(guide)


def symbolic_guided_phase_estimation() -> GuidedPhaseEstimationModel:
    """Retain the independent guide/solve instances of the source example."""
    n, k, c = symbols("n k c", positive=True, integer=True)
    guide_constraints, solve_constraints = symbols(
        "m_1 m_2", positive=True, integer=True
    )
    block, guide = _problem(n, k, c, guide_constraints, solve_constraints)
    precision, failure = symbols(r"\epsilon_{PE} \delta_{PE}", positive=True, real=True)
    return GuidedPhaseEstimationModel(
        block.system_bitsize,
        block.ancilla_bitsize,
        guide.selection_width,
        guide.junk_width,
        precision,
        failure,
        block.resource_bitsize,
    )


def symbolic_guided_hamiltonian() -> GuidedHamiltonianModel:
    """Retain the source's complementary constraint counts and block scale."""
    n, k, m, c = symbols("n k m c", positive=True, integer=True)
    zeta = symbols(r"\zeta", positive=True)
    block, guide = _problem(n, k, c, (1 - zeta) * m, zeta * m)
    threshold, gap, overlap = symbols(
        r"\lambda \alpha \gamma", positive=True, real=True
    )
    return GuidedHamiltonianModel(
        block.system_bitsize,
        block.ancilla_bitsize,
        guide.selection_width,
        guide.junk_width,
        block.alpha,
        threshold,
        gap,
        overlap,
        block.resource_bitsize,
    )
