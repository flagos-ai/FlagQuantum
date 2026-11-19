# Complete Shor workflow

| Source code cells | Native workflow |
| --- | --- |
| 4,6 | Classical random-base/gcd warm-up for 123456789; explicit local seed |
| 9 | Reuse native period finding: modular powers and inverse QFT |
| 13,15,17,19 | Exact 17/1024 convergents and all five float examples |
| 23 | Checked factoring with mathematical modular pow; explicit XOR correction |
| 26,28,29 | Original modulus21/base11, ten fractional phase bits, five residue bits and resources |
| 32,33 | Actual 2048-shot observations, count>10 and denominator<21 filters |
| 35,36 | Measured histogram and secondary rational-fraction axis |
| 38,40 | Original sixfold fraction assertion, inferred order6 and factors3/7 |

The existing native qfunc allocates zero registers, sets the residue to one,
prepares uniform phases, applies modular exponentiation and inverse QFT, and drops
the residue. Named numeric readout returns exact Fraction labels. Dropping the
entangled residue does not assert that it is zero: the full independent-state
check includes all fifteen logical bits, while only three clean scratch bits
must return to zero. Static resources are18physical bits,2116operations,depth1849;
these are instruction counts, not a hardware gate decomposition.

The source sample output establishes 2048 shots. Native seeds1700/1701 replace
unseeded hosted sampling; identical hosted samples are not claimed. Independent
Circuit probabilities sampled with the same local seed must give identical draws.
The strict source count threshold and denominator cutoff are preserved. All actual
observations remain available, including low-count approximation tails.

Two intentional mathematical improvements are disclosed. The literal source
get_factors uses Python XOR where exponentiation is intended. It coincidentally
returns3/7 for21/base11/order6, but returns1/21 for valid base8/order2. Native code
uses modular pow, checks the supplied order, and refuses trivial factors. The
source hard-codes r=6 after asserting sixfold rational labels; native code derives
r from the LCM of sampled convergent denominators, then checks that same original
assertion. Exact Fraction arithmetic implements source continued fractions;
Fraction(float) preserves the same binary input as source Rational(float).

Independent controlled modular-permutation Circuits plus a modular-function FFT
verify every complex amplitude at2e-11 without phase alignment. Clean-work leakage
must be below1e-24. Tests cover both actual seed sequences, original resources,
strict count/denominator boundaries, zero-only and invalid phases, odd orders,
trivial factors, invalid inputs and the XOR counterexamples. The large warm-up
number is used classically only; no quantum factorization of123456789 is claimed.

```sh
python -m pytest tests/integration/test_synthesis_shor_notebook.py -q
```
