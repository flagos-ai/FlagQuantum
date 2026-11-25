# MolecularData HDF5 input boundary

## Contract and ownership

An Ecosystem reader converts real closed-shell OpenFermion MolecularData HDF5
integrals immediately into the existing MolecularProblem. Algorithms, Core,
Runtime, encoding, active-space formulas and stable root exports do not change.
The reader is independent of FCIDUMP and the direct PySCF producer.

The file must contain integer scalar n_orbitals, n_electrons and multiplicity=1,
finite real scalar nuclear_repulsion, and full spatial one_body_integrals and
two_body_integrals. OpenFermion stores two_body_integrals[p,q,r,s]=(ps|qr);
the boundary explicitly converts this to native chemist (pq|rs), without a half
factor. Nuclear repulsion enters once, core_energy is zero. Optional hf_energy
is preserved; its boolean False missing-value sentinel becomes None. Reference
occupation fills the first n_electrons/2 spatial orbitals. This does not infer
orbital occupations or reproduce an electronic-structure calculation.

Only required integral/metadata datasets are loaded. Other MolecularData fields
are ignored. Complex and open-shell integrals, spin-resolved tensors, orbital
basis conversion, amplitude/RDM loading and writing are outside scope. External
or soft links and virtual datasets are rejected for consumed fields. The caller
supplies the exact path; no implicit suffix, remote access or file discovery.
The orbital budget defaults to 32 and is checked before tensor reads. Dense ERI
storage alone costs 8*NORB**4 bytes, with extra native snapshot memory required.

## Dependency and source review

Install the independent chemistry-hdf5 extra. h5py is a BSD-3-Clause HDF5 reader
maintained by the h5py project. Its NumPy dependency stops at this boundary. The
reader imports h5py lazily; native algebra, static chemistry and circuit imports
need neither h5py, NumPy, OpenFermion nor Cirq. Replacing this reader means
supplying MolecularProblem through another existing producer.

Format reference: OpenFermion commit
34adf3196cd566f4ca53893f868753786797ae2a,
src/openfermion/chem/molecular_data.py, MolecularData.save and spinorb_from_spatial.
The adapter is original code; no upstream implementation is copied.

## Acceptance

Independent format fixtures must expose the two-electron index conversion,
energy offset and all three encoding expectations. Real upstream-produced files
are checked in a separate reference environment. Missing dependencies, malformed
metadata, unsupported integrals, shape errors and allocation limits must fail
explicitly. Subprocess import isolation protects optional dependencies.

Linux testing and image qualification run in CI. Dependency metadata changes
require refreshed locked CPU images; the byte-for-byte image input checks remain
in force. Results are recorded only after execution on a named revision.
