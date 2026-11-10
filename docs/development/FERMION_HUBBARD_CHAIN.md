# Spinful Fermi-Hubbard open chain

## EXT01 stage contract

The approved EXT01 slice adds one experimental concrete constructor,
`flagquantum.algorithms.fermion.hubbard.fermi_hubbard_chain`, returning the
existing canonical `FermionOperator`. No Stable Core export or schema changes.

For L sites the Hamiltonian is

```text
H = -t sum_(i=0..L-2),s (a†_(i,s) a_(i+1,s) + h.c.)
    + U sum_i n_(i,up) n_(i,down)
    - mu sum_i (n_(i,up) + n_(i,down)).
```

Mode `2*i` is spin up and `2*i+1` is spin down. Each neighboring undirected
bond appears once. U is unshifted: there is no particle-hole or half-filling
constant. `n_sites` is a positive integer; keyword-only `t`, `u`, `mu` are
finite real numbers, defaulting to 1, 0, 0. Booleans, complex numbers and tensors
(including trainable tensors) are rejected before coercion. Invalid types raise
TypeError; nonpositive sizes and nonfinite values raise ValueError.

Assembly owns no encoding, state preparation, solver or execution policy.
Call existing `encode_fermion` with explicit `n_modes=2*n_sites`, including when
all coefficients vanish. JW, parity and BK feed the same existing measurement
consumer. Algebra constructors own anticommutation and canonicalization.
Periodic boundaries, arbitrary graphs, spinless variants, trainable model
coefficients and model registries are outside this stage. No upstream source or
new dependency is required. Existing Qmod Hubbard synthesis imports are separate.

## Scientific acceptance and contributor path

Run `python -m pytest tests/unit/fermion/test_hubbard.py -q`. Tests compare complete
small matrices with an independent occupation-bit hopping reference, Hermiticity,
spin populations and particle sectors. The two-site N=2 ground energy at mu=0 is
`(U - sqrt(U**2 + 16*t**2))/2`. At U=0, fixed-N energies are sums of the spin
pairs of open-chain levels `-2*t*cos(k*pi/(L+1)) - mu`.

Run `python -m examples.algorithms.fermion_hubbard`. For two sites, t=1, u=4,
mu=0 and N=2, each encoding measures energy -0.8284271247 and particle number 2.
The example uses bounded dense diagonalization to obtain its reference state;
its exponential inspection cost makes no many-body scalability claim. Execution
is the existing CPU `single_device_fast_path`. A fresh interpreter also exercises
this workflow with OpenFermion/Cirq and optional scientific imports blocked.
Linux default suites, packaging and version matrices run in CI.
