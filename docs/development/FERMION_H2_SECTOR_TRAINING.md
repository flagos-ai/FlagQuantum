# H2 training in an explicit molecular symmetry sector

## Scope before implementation

Algorithms owns this bounded composition of existing molecular input, native
JW encoding, candidate-restricted symmetry discovery, explicit stabilizer
reduction, ordinary circuit preparation, `fq.run` expectations and PyTorch SGD.
The scope was committed before implementation; this feature branch now
implements the paths below.
There is no new production API, solver, contract, dependency or capability entry.

Use the existing H2 STO-3G FCIDUMP fixture at 0.7414 angstrom, with two spatial
orbitals and four interleaved spin modes. Restrict discovery to the products
Z0*Z2 and Z1*Z3, then explicitly select both eigenvalues -1. For this two-orbital
register only, odd population in each two-mode spin block means exactly one
alpha and one beta electron. This implication does not hold for larger spaces.
Discovery itself must not infer these eigenvalues from the total electron count.

The reduced register has two qubits. An RY angle followed by CX prepares
cos(theta/2)|00> + sin(theta/2)|11>, corresponding to the physical determinants
cos(theta/2)|1100> + sin(theta/2)|0011>. This is a manually specified
determinant-pair ansatz for the fixed molecule, not an automatic UCCSD or general
full-register ansatz transformation. Only the float64 circuit angle is trainable.

## Independent acceptance

- Assemble an independent occupation-space reference with the existing test
  oracle, without native Pauli conversion, Clifford lowering or state lifting.
  Compare energies, analytic angle derivatives and central finite differences.
- With a=H[12,12], c=H[3,3] and b=Re(H[12,3]), the ansatz energy is
  (a+c)/2 + (a-c)*cos(theta)/2 + b*sin(theta). Compare training with its analytic
  determinant-pair minimum; separately compare that minimum with the bounded
  full N=2 spectrum for this fixture only.
- Replace FCIDUMP input with the existing native molecular snapshot while
  preserving the reduction, preparation, measurement and optimization consumer.
  Include the native energy offset and verify particle/spin populations.
- Compare angle gradients and optimizer updates with an independently embedded
  physical determinant pair. Check the selected spin parities and exact N=2
  support before and after updates. Verify reduced population observables.
- Reject a nonpreserving observable and malformed circuit angles before
  optimization. Preserve the existing refusal of trainable static-lifting inputs.
- Run the CLI in a fresh interpreter with external quantum/chemistry SDKs and
  dense eigensolvers blocked. Optimization must not prepare a dense eigenvector.

Static lifting may inspect the trained state only under `torch.no_grad()`.
It must retain its current validation and bounded CPU complex128 behavior.
Static integrals, geometry, SCF and symbolic reduction remain nondifferentiable.

## Delivery and contributor path

`examples/algorithms/fermion_h2_sector_training.py` implements the composition;
`tests/integration/test_fermion_h2_sector_training.py` owns independent scientific
scenarios. The existing algorithm example inventory tests its source and CLI,
and the fermion README links the workflow. The migration plan distinguishes the
already merged Pauli training stage from this review branch.

The shortest supported workflow and focused verification are:

```bash
python -m examples.algorithms.fermion_h2_sector_training
python -m pytest tests/integration/test_fermion_h2_sector_training.py -q
```

The recorded 120-step SGD run (initial theta=0.2, learning rate 0.2) lowers energy
from -1.0649609749 to -1.1372701747 Ha. Particle number is 2, with one electron in
each spin block. The angle changes by 0.4261362654 radians; finite-difference
gradient residual is 8.786e-13, lifted energy residual 2.220e-16 and joint-sector
residual zero. Residuals may vary across platforms.

Local review passed 66 related discovery/training scientific cases (including
the six new molecular cases) and seven selected CLI/inventory cases. The six new
cases passed again after the final formatting adjustment. Ruff, strict mypy on
the two new Python modules, Black 24.1 and 26.5, the public API baseline,
documentation consistency, the 1,418-file local-link audit and Algorithms path
ownership checks passed. These are focused checks, not a full-suite claim.
The classification is experimental CPU `single_device_fast_path`. Linux broad
checks belong to PR CI; local checks stay focused. This composition makes no
general convergence, larger-molecule, accelerator, distributed or scalability
claim and does not promote the maturity of existing scientific capabilities.
