# Spinless t-V chains and rings

## Scope and ownership

Algorithms owns two experimental concrete constructors in
`flagquantum.algorithms.fermion.hubbard`: `spinless_fermion_chain` and
`spinless_fermion_ring`. Both return the existing canonical `FermionOperator`.
Each site is one mode, with no spin duplication. Neither constructor owns an
encoding, state preparation, solver, execution policy or trainable coefficients.
There is no Stable Core export, new dependency, graph type or model registry.
Existing spinful Hubbard constructors retain their signatures and behavior.

For the set B of distinct undirected neighboring bonds:

```text
H = -t sum_(i,j in B) (a†_i a_j + a†_j a_i)
    + V sum_(i,j in B) n_i n_j - mu sum_i n_i.
```

The open chain contains bonds (0,1), ..., (L-2,L-1) and allows L >= 1.
The ring adds (L-1,0) and requires L >= 3, refusing ambiguous one/two-site cycles.
Every bond appears once, including its density interaction. V is a neighboring
interaction, not an onsite Hubbard U, and densities are unshifted: no
particle-hole offset or half-filling constant is added. Hopping has coefficient
minus t. The native algebra owns every fermionic sign; the closing hop is not
a qubit-only wraparound hop.

Keyword-only t, v and mu default to 1, 0 and 0. They must be finite static real
numbers. Bool, complex, string and tensor coefficients raise TypeError;
nonfinite coefficients raise ValueError. Site counts use the existing native
integer-index validator (including integer-index scalar types); bool and
non-indexable values raise TypeError and counts below the minimum raise
ValueError. Always encode with explicit `n_modes=n_sites`, even when all
coefficients vanish. Construction allocates no dense matrices.

## Contributor golden path

```bash
python -m examples.algorithms.fermion_spinless
python -m examples.algorithms.fermion_spinless --periodic
python -m pytest tests/unit/fermion/test_spinless.py tests/integration/test_spinless_fermion.py -q
```

At three sites, t=1, V=2, mu=0, in the two-particle sector the open chain energy
is `(V - sqrt(V**2 + 8*t**2))/2 = -0.7320508076`. The ring energy is
`V - t = 1.0` at positive t: its density interaction equals V in this sector,
and the two lowest free single-particle levels sum to -t. JW, parity and BK
all measure these energies and particle number 2 using the same measurement
consumer with explicitly mapped states. Dense diagonalization and arbitrary
state preparation are bounded exponential-cost correctness inspection.
They are not a scalable eigensolver or a general ground-state preparation path.

Change a coefficient and run the focused tests. Complete matrices for three to
five sites are compared against independent occupation-bit hopping and density
action, including negative t/V, Hermiticity and particle conservation. Free
spectra in every particle sector are sums of the one-particle levels
`-2*t*cos(k*pi/(L+1)) - mu` for chain k=1..L and
`-2*t*cos(2*pi*k/L) - mu` for ring k=0..L-1. Closing-bond signs are checked with
zero, one and two intervening spectators; the density closing bond is checked
separately. One/two-site chains, invalid inputs, all three encodings and explicit
zero-operator width are covered.

The native execution scenario composes a two-particle circuit with the static
model. Its independent expectation, analytic angle derivative, finite difference
and actual optimizer update check that interaction and closing hopping reach
execution correctly. Both examples run in a fresh interpreter while blocking
OpenFermion, Cirq, PySCF, SciPy, h5py and NumPy.

## Training golden path

```bash
python -m examples.algorithms.fermion_spinless_training
python -m examples.algorithms.fermion_spinless_training --periodic
python -m pytest tests/integration/test_fermion_spinless_training.py -q
```

The training example composes the existing `single_excitation` builder, ordinary
`fq.Circuit`, native `fq.run` expectations and PyTorch Adam. A fixed three-mode
JW register starts in |110>; completed excitations on (1,2) and (0,1) retain two
particles. Elementary decomposition gates need not individually conserve number.
Two real CPU float64 angles cover the real two-particle sector. The chain/ring
switch replaces only the static model producer; it does not alter the ansatz or
energy consumer. No dense eigenstate or domain solver initializes training.

At t=1, V=2, mu=0, 160 Adam steps with learning rate 0.08 lower chain energy
from approximately 2.3597125906 to -0.7320507169 and ring energy from 2.4576025445
to 1.0000001488, with particle number 2. Analytic N=2 energies -0.7320508076
and 1.0 are lower-bound inspection references. The fixed runs approach them;
this does not establish general convergence or a unique ring ground state.
Only circuit angles are differentiated; Hamiltonian coefficients stay static.
This is bounded CPU `single_device_fast_path` evidence, with no accelerator,
distributed, performance or capacity claim and no capability promotion.

The scenario tests compare against an independent three-state occupation matrix,
finite-difference gradients and actual optimizer steps while checking the full
state's particle-sector support. Separate tests remove the closing hop and the
closing density bond to catch sign and interaction errors independently. At
zero hopping the ring Hamiltonian is constant in this sector; its gradient and
SGD update must vanish rather than satisfy an artificial energy-descent claim.
Both workflows also run in a fresh interpreter blocking OpenFermion, Cirq,
PySCF, SciPy and h5py. Optimizer initialization may load NumPy through PyTorch;
the algebra and nontraining example retain the stronger no-NumPy isolation test.

## Limits and delivery split

Verification is bounded CPU `single_device_fast_path`. Static coefficients do
not acquire gradients because circuit angles are differentiated. Arbitrary
graphs, magnetic/twisted phases, particle-hole conventions, automatic ansatz
construction and complete upstream OpenFermion API compatibility are outside
this slice. No accelerator, distributed, performance or capacity claim follows.
Capability registration and generated limitations belong to a separate
Integration-owned PR after implementation; this domain PR does not promote
capability maturity or change the spinful model entries.
