# Native partial exponential preparation and exact amplification

Status: semantic contract; native/approved-source probe passed, independent
Circuit and formal acceptance pending. Coverage starts at 201/216.

Retain the source's five-bit increasing exponential input distribution, the
x >= 21 phase oracle, one ancilla-tuned exact amplification round and final
modulo-32 subtraction by 18. Prepare input amplitudes proportional to exp(x/4)
for rate -0.5; do not replace the algorithm with its final truncated solution.
A general classical exponential-input builder can serve later finance examples,
but those examples also require signed fractional affine arithmetic, controlled
payoff preparation and discard/reduced-channel verification before acceptance.

The original is one of the six retained semantic refusals. Its declared overlap
0.996625424765961 disagrees with the observed 0.99795457844373736. Successful
acceptance uses only the previously approved revision's overlap, retaining the
original refusal and source hashes. Verify the input distribution and marked
probability independently, and reject incorrect declared overlaps in the native
builder rather than correcting them silently.

Reuse public real-amplitude preparation, qfunc composition, numeric predicates,
phase, rotations, checked adjoint and modular in-place subtraction. A visible
attenuation port must return to zero; compiler work is checked separately.
No production API, export, ownership rule or budget change is planned.

The independent Circuit must construct product RY input preparation from
analytic bit odds, direct predicate sign matrices, inverse preparation, zero
reflection and modulo permutation. Compare complete complex amplitudes, including
absolute phase and every work/attenuation amplitude, against the approved source
and native program. Verify preparation and inverse on coherent nonzero states,
modular wrap/bit order, incorrect overlap refusal, alias rejection and budgets.
Retain original and approved resource records; report native overhead honestly.

After formal acceptance run a fresh 216-original plus approved-six replay,
archive immutable evidence and update catalog and ledger. Notebook coverage
remains separate at 24/213. Classification is `single_device_fast_path`. Do not
push, open the integration PR or run full remote CI before all 216 native
quantum-program acceptances are complete.
