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
