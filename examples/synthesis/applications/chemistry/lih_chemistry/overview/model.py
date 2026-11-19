"""Regenerate LiH at the source geometry and compare frozen-core energies."""

from dataclasses import dataclass

import torch

from flagquantum.algorithms.molecular import (
    create_molecular_hamiltonian,
    exact_ground_state_energy,
    second_quantized_hamiltonian,
)
from flagquantum.observables.fermion import FermionOperator

GEOMETRY = (("Li", (0.833472, 0.0, 0.0)), ("H", (-0.833472, 0.0, 0.0)))
BASIS = "sto-3g"
FROZEN = 1
ACTIVE_ORBITALS = 5
ACTIVE_PARTICLES = (1, 1)
WIDTH = 2 * ACTIVE_ORBITALS
SOURCE_GEOMETRY_SHA256 = (
    "a4ba935aae499837555a2c5e2696552a70c54a99ee465a76e572a4b987a7c254"
)


@dataclass(frozen=True)
class Molecule:
    orbital_energies: tuple[float, ...]
    method_energies: tuple[tuple[str, float], ...]
    fermion: FermionOperator
    active_energy: float
    native_active_energy: float


def regenerate() -> Molecule:
    """Run fresh RHF, MP2, CISD, CCSD, FCI and CASCI with optional PySCF."""
    from pyscf import ao2mo, cc, ci, fci, gto, mcscf, mp, scf

    molecule = gto.M(atom=list(GEOMETRY), basis=BASIS, unit="Angstrom", verbose=0)
    hf = scf.RHF(molecule).run()
    if not hf.converged:
        raise RuntimeError("RHF failed for the source geometry")
    mp2 = mp.MP2(hf).run()
    cisd = ci.CISD(hf).run()
    ccsd = cc.CCSD(hf).run()
    full_fci = fci.FCI(hf).run()
    for name, solver in (("CISD", cisd), ("CCSD", ccsd), ("FCI", full_fci)):
        if not solver.converged:
            raise RuntimeError(f"{name} failed for the source geometry")
    cas = mcscf.CASCI(hf, ACTIVE_ORBITALS, sum(ACTIVE_PARTICLES)).run()
    if cas.ncore != FROZEN or molecule.nelec != (2, 2):
        raise AssertionError("the source active space changed")
    one, constant = cas.get_h1eff()
    two = ao2mo.restore(1, cas.get_h2eff(), ACTIVE_ORBITALS)
    fermion = second_quantized_hamiltonian(
        torch.as_tensor(one, dtype=torch.float64),
        torch.as_tensor(two, dtype=torch.float64),
        float(constant),
    )
    native = create_molecular_hamiltonian(
        GEOMETRY,
        basis=BASIS,
        n_active_electrons=sum(ACTIVE_PARTICLES),
        n_active_orbitals=ACTIVE_ORBITALS,
    )
    native_from_integrals = exact_ground_state_energy(
        fermion, n_electrons=sum(ACTIVE_PARTICLES), n_orbitals=ACTIVE_ORBITALS
    )
    assert native.fci_energy is not None
    assert abs(native.fci_energy - cas.e_tot) < 2e-8
    assert abs(native_from_integrals - cas.e_tot) < 2e-10
    return Molecule(
        tuple(float(e) for e in hf.mo_energy),
        (
            ("RHF", float(hf.e_tot)),
            ("MP2", float(mp2.e_tot)),
            ("CISD", float(cisd.e_tot)),
            ("CCSD", float(ccsd.e_tot)),
            ("FCI", float(full_fci.e_tot)),
        ),
        fermion,
        float(cas.e_tot),
        native.fci_energy,
    )
