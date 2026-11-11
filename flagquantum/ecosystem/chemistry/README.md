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


## Populated MolecularData objects

`molecular_data.from_openfermion_molecular_data(source)` is a separate,
experimental one-way import for existing in-memory OpenFermion 1.7.1 objects.
Install the existing `openfermion` extra only in the interoperability environment.
It returns the same owned MolecularProblem as the other inputs; encoding,
active-space formulas and measurement consumers remain unchanged. No registry,
Stable Core export, dependency manifest or native algorithm is extended.

This accepts real closed-shell spatial integrals, multiplicity 1, integer
metadata, and a default positive budget of 32 spatial orbitals. Optional
`n_qubits` must equal twice the orbital count; optional `hf_energy` is retained.
The reference fills the first n_electrons/2 spatial orbitals. Nuclear repulsion
enters once and core_energy is zero. Cached upstream two-body `(ps|qr)` values
become native chemist `(pq|rs)` values without a half-factor. Numeric dense
integer/real NumPy arrays, including nonnative byte order, are converted to
owned CPU float64 tensors. MolecularProblem remains authoritative for finite
values, closed-shell electron counts and integral permutation symmetries.

Both integral arrays must already be populated. The adapter reads upstream
1.7.1's integral cache fields explicitly because its public properties may
perform implicit file reads on cache misses. Missing caches fail; this entry
never invokes `load`, `save`, a driver or an upstream calculation. Use the
existing HDF5 reader for files. Complex/open-shell/spin-orbital coefficients,
amplitudes, density matrices, arbitrary reference occupations and tensor
export are outside this boundary. Dense integral copies require O(n**4) memory
in addition to the already allocated source arrays and native snapshots.

A shortest supported import/measurement path in the optional environment:

```python
import numpy as np
import openfermion
import torch
import flagquantum as fq
from flagquantum.ecosystem.chemistry.molecular_data import from_openfermion_molecular_data
from flagquantum.algorithms.chemistry_domain.problem import fermion_hamiltonian
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.observables import to_observable

old = openfermion.MolecularData(
    [("H", (0, 0, 0)), ("H", (0, 0, 0.7))], "sto-3g", 1
)
old.n_orbitals, old.n_qubits = 2, 4
old.nuclear_repulsion = 0.5
old.one_body_integrals = np.diag([-1.0, -0.2])
old.two_body_integrals = np.zeros((2, 2, 2, 2))
old.two_body_integrals[0, 0, 0, 0] = 0.6
problem = from_openfermion_molecular_data(old)
encoded = encode_fermion(fermion_hamiltonian(problem), n_modes=4, encoding="jordan_wigner")
result = fq.run(
    fq.Circuit(4).x(0).x(1), outputs=[fq.expectation(to_observable(encoded.operator))]
)
assert abs(float(torch.as_tensor(result.expectation()).reshape(-1)[0]) + 0.9) < 1e-6
```

The illustrative integrals are supplied explicitly; this does not run RHF or
assert that they were computed for the listed geometry. For a typical change,
edit this adapter and its rejection/replacement tests, then run:

```bash
python -m pytest tests/unit/test_molecular_data_boundary.py tests/integration/test_molecular_data_isolation.py -q
python -m pytest tests/unit/test_openfermion_conversion.py -k molecular_data -q
python tools/check_architecture.py
python tools/check_dependency_policy.py
```

The native boundary tests run without NumPy or OpenFermion. Real-upstream tests
compare actual MolecularData objects and saved HDF5 files through the unchanged
native consumer, check upstream Hamiltonian matrices and all three encodings,
mutate source arrays, and forbid implicit file loading. The existing optional
workflow executes them on Python 3.10 and 3.12; fakes alone do not establish
upstream compatibility. Linux broad tiers remain CI's responsibility.

Ecosystem maintainers own this migration aid. OpenFermion remains a separately
installed Apache-2.0 dependency; its NumPy, h5py and Cirq dependencies stop at the
optional environment. No upstream source is copied. The existing optional
provider loader is reused; algorithms never import the adapter. The replacement
boundary is MolecularProblem, demonstrated by the existing HDF5 reader without
consumer changes. The adapter can be removed while experimental when in-memory
old-object migration demand ends; it introduces no permanent compatibility shim.
