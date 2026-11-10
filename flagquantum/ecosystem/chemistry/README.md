# Optional chemistry input

Owns external electronic-structure input only. The PySCF driver returns a native
MolecularProblem immediately and imports PySCF only when called. No external
object crosses into Core, algorithms or runtime. It uses no OpenFermion plugin.

Install the chemistry-pyscf extra and call
`pyscf.build_molecular_problem([("H", (0, 0, 0)), ("H", (0, 0, 0.7414))])`.
The result has four spin modes and an RHF reference energy near -1.116684 hartree.
Run `python -m pytest tests/integration/test_native_chemistry_driver.py -q`.

Dependency review: PySCF is BSD-2-Clause, maintained by the PySCF project, and is
needed for basis sets beyond the native static integral engine. Its NumPy,
SciPy and h5py objects stop here. The bounded optional distribution is resolved
by pip, never downloaded or executed by the library. Replacement/exit path:
another driver producing MolecularProblem, or the existing native driver.
Static RHF, real orbitals and fixed geometry only; no geometry-gradient claim.

## MolecularData HDF5 input

Install `flagquantum[chemistry-hdf5]` and call
`molecular_hdf5.read_molecular_hdf5("molecule.hdf5")`. This independent reader
requires no PySCF, OpenFermion or Cirq. It returns the same MolecularProblem as
the direct driver. Real closed-shell spatial integrals only; physicist-order
file integrals become native chemist order. The nuclear constant enters once.
Optional HF energy is preserved, and a missing boolean sentinel becomes None.
The default orbital budget is 32 before dense tensor reads. The reference fills
the first NELEC/2 orbitals; arbitrary occupations are not inferred from the file.
No amplitudes, density matrices, complex/open-shell data or writing are supported.
See the [format contract](../../../docs/development/MOLECULAR_HDF5_INPUT_PR.md).
