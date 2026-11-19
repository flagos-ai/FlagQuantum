# Source workflow and acceptance

| Source code cell | Native counterpart |
| --- | --- |
| 2: size 6, maximum width 12, CX objective | Fixed six-bit type; synthesis width ceiling 12; actual CX-cost selection |
| 3: coin-conditioned increment/decrement | Native qfunc arithmetic helpers and coherent controls; equivalent Fourier candidate |
| 4: CX/U basis and custom transpilation | Explicit caller-supplied immutable exact-rule extension; actual converted IR |
| 6: synthesis time, width, depth and CX | Per-candidate synthesis/conversion and whole-workflow duration; same-basis resource estimator |

The compiler's shared defaults and public APIs are unchanged. Explicit H, phase,
controlled phase, X, T and inverse-T identities target U3; existing SWAP and CCX
rules are reused. All conversion preserves absolute phase, metadata and trainable
angle gradients. Conversion is bounded to 20,000 added operations; no matrix
fallback or unreported basis substitution is used.

Acceptance checks all 128 logical input columns for both native and converted
candidates against the independent source transition, plus a complex coherent
input and an independently constructed Circuit. This covers wraparound, bit
order, coin-dependent sign, absolute phase and work reset. Seventeen elementary
operator identities are checked on every column; phase gradients and metadata
are checked separately. Width, operation, depth, memory and conversion budgets
must refuse undersized requests. The Notebook's inline qfunc is executed and
compared independently, in addition to CLI and actual Jupyter execution.

Candidate selection minimizes actual CX among the two visited candidates, with
deterministic depth/width/name ties. It is not global optimization and is not a
comparison with Classiq's private search or compile time. Local execution is
`single_device_fast_path`; there is no accelerator or speedup claim. Source
Notebook has no sampling or plotting stage. Source outputs remain empty.
