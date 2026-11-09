# Static molecular problems

Owns real spatial-orbital integral semantics, explicit energy offsets and
closed-shell active-space transformations. Dependencies are native algebra,
existing molecular assembly and PyTorch. No external chemistry driver, runtime
or device selection belongs here. No new integral or SCF engine is implemented.

Use problem.from_native_molecule with an existing
algorithms.molecular.create_molecular_hamiltonian result, then
problem.fermion_hamiltonian and fermion.encoding.encode_fermion. Offsets enter
the fermionic Hamiltonian once; never add energy_offset to its measured energy.

Change an active selection and run
`python -m pytest tests/unit/fermion/test_chemistry.py -q`. Tensor inputs are
owned static snapshots. Geometry and integral differentiation are unsupported.
