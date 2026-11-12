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

## Input replacement during circuit training

Native integrals, `fcidump.read_fcidump` and
`molecular_hdf5.read_molecular_hdf5` all produce `MolecularProblem`. The training
consumer uses its static integrals, electron count and explicit occupation
convention; it does not need an external file object or optional reference
energy. For a two-orbital, two-electron problem, continue the native workflow
above with this existing UCCSD/PyTorch composition:

```python
import torch
import flagquantum as fq
from flagquantum.algorithms.chemistry import uccsd_ansatz
from flagquantum.algorithms.chemistry_domain.problem import fermion_hamiltonian
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.observables import to_observable

encoded = encode_fermion(fermion_hamiltonian(problem), n_modes=4, encoding="jordan_wigner")
observable = to_observable(encoded.operator)
angles = torch.tensor([0.1, -0.07, 0.2], dtype=torch.float32, requires_grad=True)
optimizer = torch.optim.Adam([angles], lr=0.05)
for _ in range(120):
    optimizer.zero_grad()
    circuit = uccsd_ansatz(4, 2, angles, dtype=torch.complex128, device="cpu")
    loss = fq.run(circuit, outputs=[fq.expectation(observable)]).expectation().sum()
    loss.backward()
    optimizer.step()
```

Replacing only the `problem` producer with either file reader preserves the
consumer. The H2 scenario at 0.7414 angstrom reaches approximately
-1.1372701747 Ha with one particle of each spin. FCIDUMP's undivided constant
and HDF5's nuclear constant enter once through native assembly. Missing optional
HF energies do not change the objective. UCCSD angle ingestion remains float32,
even with a complex128 state; this evidence is a bounded CPU/full-JW trajectory,
not general convergence, geometry/SCF gradients or scalable training.

With `flagquantum[chemistry-hdf5]` installed, run
`python -m pytest tests/integration/test_molecular_hdf5_isolation.py -k training -q`.
It compares all 121 energy evaluations, initial gradients and optimized angles
after replacing the input producer, then checks final spin-sector support.
The complete scenario blocks OpenFermion, Cirq, PySCF and SciPy imports in a fresh
interpreter; h5py and NumPy are confined to file production/reading and PyTorch's
existing optimizer initialization. The same test is selected by the existing
optional HDF5 CI lane; no new dependency, training API or runtime is introduced.
