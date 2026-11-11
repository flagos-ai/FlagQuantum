# Molecular data and orbital rotations

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
owned static snapshots. Geometry and supplied-integral differentiation are unsupported.

`orbital_rotation.rotate_spatial_integrals` separately transforms static real
integrals with trainable real orbital parameters. Its tensor results preserve
autograd and must not be passed to static `MolecularProblem` snapshots while
tracking gradients. `closed_shell_determinant_energy` contracts those tensors
for explicit doubly occupied orbitals, including the supplied constant once.
Run `python -m examples.algorithms.fermion_orbital_rotation` for the orbital
gradient and one descent step; test with
`python -m pytest tests/unit/fermion/test_orbital_rotation.py -q`.
See [the stage contract](../../../docs/development/FERMION_ORBITAL_GRADIENTS.md)
for conventions and limits.

Shortest executable workflow:

```python
from flagquantum.algorithms.molecular import create_molecular_hamiltonian
from flagquantum.algorithms.chemistry_domain.problem import from_native_molecule
from flagquantum.algorithms.chemistry_domain.active_space import select_active_space

molecule = create_molecular_hamiltonian([("H", (0, 0, 0)), ("H", (0, 0, 0.7414))])
problem = from_native_molecule(molecule)
active = select_active_space(problem, active_orbitals=(0, 1)).problem
assert active.n_modes == 4
```

External driver replacement requires changing only the producer: pass a
MolecularProblem returned by ecosystem.chemistry.pyscf.build_molecular_problem to
the same active-space, encoding and energy consumers. Orbital labels survive
selection, and reference_occupied records occupations after arbitrary reordering.
