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

## FCIDUMP input

`fcidump.read_fcidump(path)` reads real restricted closed-shell spatial integrals
directly into the same MolecularProblem, using only the standard library and
PyTorch. It accepts NORB, NELEC, MS2=0, optional IUHF=0, ORBSYM and ISYM;
`&END` or `/` terminates the header. Integral indices are one-based chemist
notation, with eightfold ERI symmetry. Missing records are zero. Fortran D
exponents, blank lines and `!` comments are supported. Conflicting symmetry
duplicates, unknown headers, orbital-energy records, complex integrals and
unrestricted blocks fail explicitly. Symmetry labels are checked but do not
select a reference, perform tapering or become native orbital labels.

The undivided file constant becomes core_energy, with nuclear_repulsion=0, so
the native Hamiltonian includes it exactly once. The default reference fills the
first NELEC/2 orbitals and carries no HF-energy claim. The default max_orbitals=32
guards dense O(NORB**4) storage; callers must explicitly raise the allocation
budget for larger inputs. This parser is original format-boundary code, not
copied from an upstream implementation. It introduces no writer, SCF engine,
format registry, dependency extra or stable root export.

Run `python -m examples.algorithms.fermion_fcidump` for three encodings of a
two-orbital input, each giving -0.9 Ha. Run
`python -m pytest tests/unit/fermion/test_fcidump.py -q` when changing record
validation. Optional independent producer checks use PySCF-written H2 STO-3G and
6-31G files in `tests/integration/test_fcidump_pyscf.py`; PySCF is never imported
by the reader. Active-space and encoded measurement consumers remain unchanged.
