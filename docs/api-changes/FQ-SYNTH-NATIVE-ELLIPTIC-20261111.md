# Native elliptic-curve coherent scalar addition

Status: contract and feasibility probe; not accepted. Coverage remains 205/216.
Classification: `single_device_fast_path`.

Preserve the original curve y^2=x^3+5x+4 over F_7, generator (0,5), two
three-bit uniform scalar registers, seed (4,2), six-bit point encoding and both
inverse Fourier transforms. Validate the second original accepted seed (3,2)
as well. The original closed importer proves the reachable affine orbit and
rejects dirty workspace or relative phases. Native acceptance must independently
preserve those states and all absolute phases.

Use classical field arithmetic to construct complete point-translation
permutations on every curve point, including an explicit infinity sentinel.
Extend the permutation by identity on invalid six-bit words. This is a bounded
reversible primitive derived from the curve, not a compiled final state or
lookup from source execution. Decompose each permutation into Gray-path basis
transpositions. Native qfunc composes swaps, masked array controls, within/apply,
coherent scalar controls and inverse QFT. Do not raise the static matrix limit
or add an opaque six-qubit matrix to bypass its current five-qubit boundary.

The infinity sentinel is (7,7), outside the finite field; source frozen inputs
stay in their proven nonsingular cosets. Document the native extension rather
than claiming the source defines all exceptional affine branches. Keep scalar
raw words and fractional /8 interpretation explicit, and preserve point x/y
field ordering in full-state comparisons.

Acceptance requires both source seeds, at least one coherent nonuniform scalar
input, every point-translation basis word and inverse, absolute-phase
interference, hidden-work cleanup, alias rejection and resource budgets.
Independent Circuit uses direct field group arithmetic and a separate scalar
sum/output XOR construction with analytic inverse DFT, rather than the native
Gray decomposition. Original bytes and resource baseline remain unchanged.
Any additional native workspace is measured and verified clean.

After acceptance, run fresh 216-original and approved-six replay, archive
immutable evidence, regenerate catalog and update delivery ledger. Notebook
workflow acceptance stays separate. No push, integration PR or full remote CI
before all 216 native quantum programs have independent acceptance.
