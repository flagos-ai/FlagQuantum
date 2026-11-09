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
