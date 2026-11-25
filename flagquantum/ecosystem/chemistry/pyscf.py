"""Direct, explicitly optional PySCF RHF boundary; no OpenFermion intermediate."""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence

import torch

from ...algorithms.chemistry_domain.problem import MolecularProblem


def build_molecular_problem(
    geometry: Sequence[tuple[str, Sequence[float]]],
    *,
    basis: str = "sto-3g",
    charge: int = 0,
    convergence_tolerance: float = 1e-10,
    max_cycles: int = 100,
) -> MolecularProblem:
    """Run static closed-shell RHF and return real canonical MO integrals.

    Geometry is in angstrom; energies and integrals are in hartree. No network,
    credential or remote execution is used. Install the chemistry-pyscf extra.
    """
    if isinstance(charge, bool):
        raise TypeError("charge must be an integer")
    charge = operator.index(charge)
    if isinstance(max_cycles, bool) or operator.index(max_cycles) < 1:
        raise ValueError("max_cycles must be a positive integer")
    max_cycles = operator.index(max_cycles)
    if not math.isfinite(convergence_tolerance) or convergence_tolerance <= 0:
        raise ValueError("convergence_tolerance must be finite and positive")
    if not isinstance(basis, str) or not basis.strip():
        raise ValueError("basis must be a nonempty basis-set name")
    atoms = []
    for symbol, coordinates in geometry:
        if not isinstance(symbol, str) or not symbol.strip() or len(coordinates) != 3:
            raise ValueError("geometry requires element names and three coordinates")
        if any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            for x in coordinates
        ):
            raise ValueError("coordinates must be finite static real numbers")
        atoms.append((symbol, tuple(float(x) for x in coordinates)))
    if not atoms:
        raise ValueError("geometry must contain at least one atom")
    try:
        from pyscf import ao2mo, gto, scf
    except ImportError as exc:
        raise ImportError(
            "install flagquantum[chemistry-pyscf] for the direct PySCF driver"
        ) from exc
    molecule = gto.M(
        atom=atoms, basis=basis, charge=charge, spin=0, unit="Angstrom", verbose=0
    )
    if molecule.nelectron % 2:
        raise ValueError("this driver supports even-electron closed-shell RHF only")
    solve = scf.RHF(molecule)
    solve.conv_tol = convergence_tolerance
    solve.max_cycle = max_cycles
    energy = float(solve.kernel())
    if not solve.converged:
        raise ValueError("PySCF RHF did not converge within the requested cycles")
    coefficients = solve.mo_coeff
    n = coefficients.shape[1]
    one = torch.tensor(
        coefficients.T @ solve.get_hcore() @ coefficients, dtype=torch.float64
    )
    two = torch.tensor(
        ao2mo.kernel(molecule, coefficients, compact=False).reshape(n, n, n, n),
        dtype=torch.float64,
    )
    return MolecularProblem(
        one,
        two,
        int(molecule.nelectron),
        float(molecule.energy_nuc()),
        reference_energy=energy,
    )
