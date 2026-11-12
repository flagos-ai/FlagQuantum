# Spinful Fermi-Hubbard periodic ring

## Scope and ownership

The experimental concrete constructor
`flagquantum.algorithms.fermion.hubbard.fermi_hubbard_ring` returns the existing
canonical `FermionOperator`. It reuses `fermi_hubbard_chain` and adds only the
closing bond. The open-chain signature and behavior remain unchanged. No Stable
Core export, external dependency, solver, topology registry or new value type is
introduced. Algorithms owns symbolic model assembly; encoding and execution stay
with their existing consumers.

For L >= 3 sites, with site indices taken modulo L:

```text
H = -t sum_(i=0..L-1),s (a†_(i,s) a_((i+1) mod L,s) + h.c.)
    + U sum_i n_(i,up) n_(i,down)
    - mu sum_i (n_(i,up) + n_(i,down)).
```

Each undirected bond is counted once, including the closing bond between L-1
and 0. Interleaved modes 2*i and 2*i+1 carry up/down spin; U is unshifted and no
half-filling constant is added. An even number of intervening occupied modes
gives hopping matrix element -t, and an odd number gives +t. The native algebra
owns that sign; there is no special Pauli boundary string in the model builder.

Keyword-only t, u and mu default to 1, 0 and 0 and must be finite static real
numbers. Booleans, complex values and tensors raise TypeError. A noninteger site
count raises TypeError; fewer than three sites or a nonfinite coefficient raises
ValueError. One/two-site cycles are refused rather than silently choosing self
bonds or duplicated parallel edges. Use the existing chain constructor when an
isolated site or one two-site bond is intended. This convention does not claim
full upstream OpenFermion model API compatibility.

## Contributor golden path

Run the existing measurement consumer with only its model producer switched:

```bash
python -m examples.algorithms.fermion_hubbard --periodic
python -m pytest tests/unit/fermion/test_hubbard.py tests/unit/fermion/test_hubbard_ring.py tests/integration/test_fermion_hubbard_ring.py -q
```

The bounded three-site example sets t=1, U=4 and mu=0. In the two-particle
sector its ground energy is `(U - 2*t - sqrt((U + 2*t)**2 + 32*t**2))/2`,
namely -3.1231056256 at the stated positive t. JW, parity and BK all measure
that energy and particle number 2 after mapping the reference state to the same
encoding. The example still defaults to the original two-site open-chain path.
Its dense diagonalization and arbitrary-state preparation have exponential cost;
they are correctness inspection, not a scalable model solver.

Change t, U or mu in a small model and run the focused tests. The independent
occupation-bit oracle checks the complete action for three/four sites, including
Hermiticity and separate spin conservation. Free-particle spectra in every
particle sector are compared with sums of spin-degenerate periodic levels
`-2*t*cos(2*pi*k/L) - mu`. Explicit spectator occupations protect the closing
bond's fermionic sign, and all three encodings are checked in their occupation
bases. A separate native execution scenario checks an analytic expectation,
angle gradient, finite difference and one optimizer update with a spectator on
the closing hop. A fresh interpreter runs the periodic example while blocking
OpenFermion, Cirq, PySCF, SciPy, h5py and NumPy.

## Limits

The constructor is symbolic and retains explicit encoding width `2*n_sites`,
including the zero operator. It does not own state preparation or training.
Coefficient gradients, spinless models, arbitrary graphs, twisted boundary
phases and magnetic fields are unsupported. Verification is bounded CPU
`single_device_fast_path`; no accelerator, distributed, performance or capacity
claim follows. Circuit angle differentiation in the composed integration test
does not make static Hamiltonian coefficients trainable. The open-chain
capability entry remains specific to the chain. Ring catalog registration is
owned separately by Integration; this slice does not promote capability maturity.

## Training golden path

```bash
python -m examples.algorithms.fermion_hubbard_ring_training
python -m pytest tests/integration/test_fermion_hubbard_ring_training.py -q
```

The fixed three-site, two-electron example composes the existing ring producer,
full JW observable, six-mode UCCSD circuit, native `fq.run` and PyTorch Adam.
It prepares one particle of each spin without a dense eigenvector and runs 60
parameter updates. The analytic ground energy -3.1231056256 is a lower-bound
inspection reference, not a promised optimizer result. UCCSD normalizes angles
to float32; the example reports central differences at epsilon=1e-3, parameter
movement and separate final spin populations. Optimizer initialization may import
NumPy through PyTorch; OpenFermion, Cirq, PySCF, SciPy and h5py are unnecessary.

Change the initial angles and run the focused tests. A separate nine-dimensional
distinguishable-particle lattice Hamiltonian checks arbitrary nonzero UCCSD
energies with the interleaved-occupation sign applied explicitly. Central
differences of that reference check native gradients; an optimizer update must
lower energy and retain one up and one down particle. Replacing the ring producer
with the chain must change both energy and gradient in the same consumer.
Incorrect parameter counts fail at the existing ansatz boundary. This composition
owns no new algorithm or execution contract, does not differentiate static model
coefficients and makes no general convergence, accelerator or scalability claim.
