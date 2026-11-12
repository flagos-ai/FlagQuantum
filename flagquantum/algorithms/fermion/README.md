# Native fermionic science

Owns operator encodings, static decompositions and circuit synthesis. Reuses the
existing public FermionOperator. Core Pauli values contain no execution objects.
Observable conversion lives here, not in Core. Import concrete modules instead
of expanding the stable root API.

Dependencies: standard library and PyTorch. This domain must not import Cirq,
OpenFermion, external chemistry drivers or runtime internals. Consumers execute
ordinary circuits through fq.run. Static preprocessing refuses autograd inputs;
time-dependent circuit builders preserve their declared tensor parameters.

Run `python -m examples.algorithms.fermion_energy` to see the same measurement
consumer use JW, parity and BK, each reporting 1.3. Modify a number-operator
coefficient and run `python -m pytest tests/unit/fermion -q` to verify a change.

Dense matrix helpers are explicitly bounded inspection tools. They do not
establish large-system capacity, distributed training or accelerator support.

Modules are split by numerical responsibility: decomposition.py returns adjacent
orbital matrices without importing Circuit; lowering.py lifts them to JW gates;
preparation.py creates Slater determinants and canonical Gaussian vacua.
low_rank.py owns signed spatial ERI factors and their contraction correction.
evolution.py composes static orbital bases and differentiable density phases.
swap_network.py owns FSWAP signs and final layout independently of evolution.
symmetry.py accepts explicit encoded Z parities and returns the state embedding.
No execution backend, adapter registry or extra general-purpose IR is introduced.

`python -m examples.algorithms.fermion_chemistry` is the ten-minute chemistry,
encoding, measurement and differentiable-time example. It requires only the core
install, prints -1.1166843871 Ha for each encoding, and runs on CPU. To change a
Givens convention, first run test_preparation.py: every particle sector is checked
against independent complex minors. For factor conventions, run test_evolution.py:
Fock matrices include the contraction and exact offset phase. Circuit synthesis
uses unreduced JW registers only. Density-pair gates may be nonadjacent.

Fourier transforms use fourier.py for radix-2 adjacent orbital factors and
lowering.fermionic_fourier_transform for the full JW circuit. The mathematical
plan reuses OrbitalDecomposition and imports no circuit or execution code.
Forward F[p,q] = exp(-2j*pi*p*q/n)/sqrt(n); inverse uses F dagger. Both retain
natural mode order, fix the vacuum phase to one, and preserve every fermionic
determinant sign. Supported widths are positive powers of two, including one.
This is a fermionic basis change, not the qubit QFT. Static CPU complex128 only;
parity/BK/tapered registers, trainable Fourier factors and performance claims are
outside this profile. Radix-2 butterflies need O(n*log(n)) interactions before
routing; this adjacent implementation stores and emits O(n**2) factors.

Run `python -m examples.algorithms.fermion_fourier`: a particle initially in mode
one has occupation 0.25 in each of four output modes, and the inverse restores
its amplitudes below 1e-12. Run `python -m pytest tests/unit/fermion/test_fourier.py
-q` to check plans against torch.fft and complete small Fock-space matrices
against independent Fourier minors, without phase alignment.

Explicit Pauli stabilizer reduction lives in stabilizer.py. It accepts independent
commuting signed Pauli words and explicit eigenvalues in the operator's supplied
qubit basis. It does not discover symmetries or infer fermionic mode labels.
StabilizerReduction combines the reduced operator, existing native Clifford
Instruction values and the existing ReductionMap in the diagonal frame. Use
lowering.lift_stabilizer_state for the original-register embedding C dagger J;
mapping alone is only J. This distinction matters for X/Y symmetry eigenstates.
reduce_operator reuses that frame for another preserving observable and refuses
nonpreserving ones. Existing reduce_z_parities retains its interface and affine
mapping semantics. Both paths use clifford.py's exact CX Pauli conjugation.

Symbolic reduction allocates no dense arrays; state lifting is an explicit CPU
complex128 inspection operation limited to 20 full qubits. Static state inputs
must be finite and cannot require autograd. Zero remaining qubits are supported
as a scalar operator and one-element reduced vector. Clifford construction uses
ordinary H, S dagger and CX gates and makes no hardware-connectivity claim.
There is no automatic tapering of an ansatz, accelerator/distributed execution
claim, optimizer or eigensolver in this feature.

Run `python -m examples.algorithms.fermion_tapering`: a fixed four-qubit H2
problem is reduced to two qubits, measured at -1.1372701747 Ha with two particles,
and its state lifted with a residual below 1e-12. The example explicitly supplies
the sector and obtains the state by small dense diagonalization, then uses the
existing state-preparation routine and fq.run. Run
`python -m pytest tests/unit/fermion/test_stabilizer.py -q` for independent dense
projectors, complete sector identities, signed/random Clifford frames and input
validation. The stage contract is
[explicit stabilizer reduction](../../../docs/development/FERMION_STABILIZER_REDUCTION.md).
Controlled time evolution reuses `low_rank_evolution(...).control(1,
ctrl_qubits=2*n_modes)`: it is not a separate algorithm or public wrapper. The
control branch retains the exact energy-offset phase, while both branches use
the existing product formula. Only its scalar time supports autograd. Run
`python -m examples.algorithms.fermion_controlled_evolution` for an H2 control
interferometer and `python -m pytest tests/unit/fermion/test_controlled_evolution.py
-q` for independent full-branch, phase, convergence and gradient references.
 That profile is unreduced JW/CPU only and makes no routing or hardware claim.

For file-to-training composition, run
`python -m examples.algorithms.fermion_fcidump_training`. The example composes
the existing FCIDUMP adapter, fermionic Hamiltonian assembly, full JW encoding,
UCCSD circuit and native runtime with PyTorch Adam. It accepts only two orbitals
and two electrons and uses a small dense N=2 energy as an inspection reference.
It owns no parser, ansatz or solver implementation. UCCSD angles remain float32;
geometry/SCF gradients and scalable optimization are not implied. Run
`python -m pytest tests/integration/test_fermion_fcidump_training.py -q` to check
file/native producer replacement, gradients, particle-sector support, invalid
inputs and execution without external quantum or chemistry SDKs.

For the delivered stages, file-input workflows and planned extension boundaries,
see the [migration status](../../../docs/development/OPENFERMION_NATIVE_MIGRATION.md#current-delivery-status).
The initial PR description and its CI counts are historical records; the status
section identifies the revision to which its delivery table applies.

Spinful open-chain model assembly lives in `hubbard.py` and returns the existing
`FermionOperator`. It owns only the static Hamiltonian: interleaved up/down modes,
minus-t nearest-neighbor hopping, unshifted U double occupancy and minus-mu
population. It must not own encoding, state preparation, runtime policy or a
solver. Dependencies stop at native scalar validation and fermionic algebra;
OpenFermion, Cirq and additional science packages are unnecessary.

Run `python -m examples.algorithms.fermion_hubbard` for the ten-minute model path:
build two sites with t=1, u=4, mu=0, select N=2 and measure -0.8284271247 with
particle number 2 through JW, parity and BK. Change U and compare the analytic
energy printed by the example with its measured values. Dense reference state
construction is an exponential inspection tool, not a scalable eigensolver.
Run `python -m pytest tests/unit/fermion/test_hubbard.py -q` for independent
occupation-bit and analytic sector references before changing assembly. The
[EXT01 contract](../../../docs/development/FERMION_HUBBARD_CHAIN.md) records
validation and unsupported boundaries. Always encode with explicit
`n_modes=2*n_sites`, including the zero Hamiltonian.

For a training composition, run
`python -m examples.algorithms.fermion_hubbard_training`. It uses the same static
Hamiltonian with ordinary circuit angles, native CPU expectations and PyTorch
Adam; it adds no domain solver or trainable Hamiltonian coefficients. Its
four-mode JW preparation keeps one particle of each spin after each completed
preparation, rather than after every gate. Change the initial angles and run
`python -m pytest tests/integration/test_fermion_hubbard_training.py -q` to check
sector support, finite-difference gradients and an optimizer update. The analytic
two-site energy is an inspection reference, not a many-body performance claim.

The same `hubbard.py` module also owns the experimental static
`fermi_hubbard_ring` constructor for periodic cycles of at least three sites. It
reuses the chain assembly and adds one closing bond per spin; native algebra
retains the occupation-dependent fermionic sign. Run the existing consumer with
`python -m examples.algorithms.fermion_hubbard --periodic` for a three-site N=2
reference energy of -3.1231056256 at t=1, U=4, mu=0 through JW, parity and BK.
See the [ring scope and contributor path](../../../docs/development/FERMION_HUBBARD_RING.md)
and run `python -m pytest tests/unit/fermion/test_hubbard_ring.py
tests/integration/test_fermion_hubbard_ring.py -q` after changing this behavior.
One/two-site cycles are refused, coefficients remain static, and the dense
reference remains a bounded correctness oracle.

For the bounded periodic training composition, run
`python -m examples.algorithms.fermion_hubbard_ring_training`. Existing six-mode
UCCSD and PyTorch Adam optimize circuit angles for the three-site ring with
one up and one down particle. No dense reference state, domain solver or runtime
change is needed. Run
`python -m pytest tests/integration/test_fermion_hubbard_ring_training.py -q`
for independent lattice energies, finite-difference gradients, spin-sector
preservation, closing-bond sensitivity and dependency isolation. See the
[training golden path](../../../docs/development/FERMION_HUBBARD_RING.md#training-golden-path)
for float32 angle precision and bounded CPU scope.

The same model module owns experimental `spinless_fermion_chain` and
`spinless_fermion_ring`: one mode per site, minus-t hopping, unshifted
nearest-neighbor V interaction and minus-mu population. They return existing
fermion values and own no solver or runtime. Open chains allow one site; rings
require at least three. Static validation reuses the existing scalar and count
boundaries. Encode with explicit `n_modes=n_sites`, including zero operators.
Run `python -m examples.algorithms.fermion_spinless` and add `--periodic` to
replace only the model producer; three encodings measure -0.7320508076 or 1.0
respectively at t=1, V=2, mu=0 and N=2. See the
[spinless golden path](../../../docs/development/FERMION_SPINLESS.md) and run
`python -m pytest tests/unit/fermion/test_spinless.py tests/integration/test_spinless_fermion.py -q`
for independent occupation action, boundary signs, free spectra, density bonds,
encodings and native gradients. Dense state preparation remains a bounded CPU
inspection tool; no coefficient-gradient or scalability claim is introduced.

For spinless training, run
`python -m examples.algorithms.fermion_spinless_training` and add `--periodic`
to replace only the static producer. Existing single excitations prepare N=2 in
three full JW modes, and native expectations feed 160 PyTorch Adam steps on two
float64 circuit angles. Change the initial angles and run
`python -m pytest tests/integration/test_fermion_spinless_training.py -q` to
check occupation-matrix energies, finite-difference gradients, closing bonds,
particle-sector support and native dependency isolation. See the
[training golden path](../../../docs/development/FERMION_SPINLESS.md#training-golden-path)
for expected values and the zero-hopping stationary case. This example adds no
solver or model API and makes no coefficient-gradient or scalability claim.


Diagonal symmetry discovery lives in `symmetry.discover_z_parities`. It returns
independent encoded Z words for the existing `reduce_z_parities` consumer;
optional independent candidate words restrict the search to their product span.
It selects no eigenvalues and does not infer particle-number conservation.
Run `python -m examples.algorithms.fermion_parity_discovery` for the shortest
explicit-total-parity path, then change a hopping term and run
`python -m pytest tests/unit/fermion/test_parity_discovery.py tests/integration/test_fermion_parity_discovery.py -q`.
See the [discovery scope](../../../docs/development/FERMION_PARITY_DISCOVERY.md)
for encoded-index conventions, exact coefficients and unsupported X/Y discovery.


## Explicit parity-sector training composition

Run `python -m examples.algorithms.fermion_parity_training` for the ten-minute
path from a fixed pairing Hamiltonian through total-parity discovery, explicit
+1 selection, reduced-register angle training and lifted-state inspection.
Change the static pairing coefficient and run
`python -m pytest tests/integration/test_fermion_parity_training.py -q`.
The tests replace JW with parity and BK without changing the training consumer,
and check energies and gradients against an independent occupation-space matrix.
Symbolic discovery/reduction remain static: they own no optimizer, runtime or
trainable-coefficient conversion. This experimental CPU example trains one
circuit angle, not model coefficients or an automatic matched tapered ansatz.
The selected sector contains N=0 and N=2; it is not an exact-number projection.
