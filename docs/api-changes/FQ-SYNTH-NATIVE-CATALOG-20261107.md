# Native catalog gate and preparation admission

The user authorized completing corresponding Python qfunc and independent
Circuit examples for the pinned 216-case catalog, implementing missing native
operations where needed. This unreleased successor admits eight operations in
`flagquantum.synth`; it does not change Stable Core exports, existing operation
signatures, `synthesize`, execution, serialization or historical admission files.
The exact additive signatures are recorded in
[the candidate contract](../../contracts/synthesis-native-catalog-candidate.json).
Release authorization remains false.

## Semantics and reuse

`X`, `RZ`, `PHASE` and `RZZ` retain the Qmod gate conventions. RZ is
`diag(exp(-i*a/2), exp(i*a/2))`; PHASE is `diag(1, exp(i*a))`.
RZZ is `exp(-i*a*Z*Z/2)` on exactly two bits. Angles are finite static reals
in radians. The existing register gates, two-bit RZZ operation, initialized
storage checks and lowering own their implementation. Wider scalar targets,
uninitialized targets and unsupported host expressions are refused.
`synth.X` is a captured gate; the existing Stable Core `fq.X` observable
constructor retains its identity and signature.

`prepare_bell_state(1, target)` prepares Phi-, and state 2 prepares Psi+.
`prepare_ghz_state(size, target)` prepares a fresh 1..256-bit GHZ register.
`prepare_amplitudes(vector, bound, target)` accepts a normalized finite real
literal list or tuple of power-of-two length 2..256. `prepare_state` accepts
nonnegative probabilities summing to one within absolute tolerance 1e-12 and
prepares their nonnegative square roots. Normalization is checked, never repaired.
The bound must be a finite nonnegative static real. The existing realization is
exact; probability preparation uses bound zero internally, as the Qmod path does.
Literal fractions require a positive integer denominator. Capturing these calls
never evaluates an arbitrary host call or executes a qfunc body.

Preparation owns fresh storage and infers its width. Preallocation, repeated
initialization, mutation of Const operands and mismatched declared widths are
rejected by the existing register verifier. These operations lower directly to
`qreg.prepare_ghz`, `qreg.prepare_amplitudes` and `qreg.gate`; there is no Qmod
round trip or new state preparation algorithm. A native QNum groups the logical
bits; general Python quantum arrays/views and runtime classical binding remain
outside this slice.
Direct gate calls preserve the current refusal to mutate Const operands.
The existing `phase` expression provides diagonal phase composition through
Const helpers and coherent control; admitting mutable helper borrows is a
separate native-language extension, not an implicit change in this batch.

## Runnable user workflow

```python
import flagquantum as fq
from flagquantum.synth import Output, QNum, prepare_bell_state, qfunc, synthesize

@qfunc
def main(x: Output[QNum]) -> None:
    prepare_bell_state(2, x)

program = synthesize(main)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x",))))
```

The named distribution is approximately `{(1,): 0.5, (2,): 0.5}` with
positive amplitudes. The corresponding catalog entry retains the exact Qmod
original and an independently constructed Circuit comparison.

## Acceptance

Run corresponding original Qmod, native qfunc and independent Circuit cases.
Compare every complex amplitude without global-phase alignment; check register
order, zero workspace, state preparation bounds and resource rejection. Include
coherent gate probes that distinguish RZ from PHASE and preparation probes with
negative amplitudes. Retain all original source blobs and AO/PF resource records.
Keep catalog-native coverage distinct from Qmod execution smoke coverage and
links to existing Circuit validations.
