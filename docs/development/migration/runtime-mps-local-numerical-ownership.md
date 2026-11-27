# Runtime MPS local numerical ownership

Status: formal source and native installed artifact qualified after parent #1339
`6ea206c3b4aa82fe73f33e8515f018d2fef7a1ec` passed its exact-head combined CI.
This child retains its own exact published-head CI requirement.

## Numerical data flow

Five private functions join the existing Simulation JAX MPS pullback module.
The NumPy kernels retain CPU development execution without requiring JAX.
The original CPU path already imports Torch through Runtime backend policy;
this change introduces neither a new dependency nor a new package.

| Runtime donor | Simulation implementation | Retained responsibility |
| --- | --- | --- |
| `mps/backward.py`, nested rank step | `_jax_mps_ry_site_response` | Device admission, pmap, ppermute and warmup/synchronization |
| `mps/backward.py`, CPU branch | `_numpy_mps_ry_site_response`, `_numpy_mps_ry_weighted_gradient` | Two explicit rank copies, event timers and evidence assembly |
| `mps/canonicalization.py`, optimizer rank step | `_jax_mps_ry_z_response` | Remote contribution, SGD policy and owner-local writeback |
| `mps/pullbacks.py`, owner-rank VJP | `_numpy_mps_owner_local_scores` | Cotangent exchange, actual local VJP invocation and result ownership |

The site-contraction response and direct cosine response remain separate:
replacing the former by a trigonometric identity could change rounding.
The optimizer now evaluates its pure local derivative alongside its score,
before the same ppermute. No timer, collective order, random draw or policy
moves. Numerical expressions, dtypes and operations retain their original form.

## Deliberate Runtime stopping point

The RY/RXX local tensors inside the owner-rank VJP supply only memory witness
shape, dtype and bytes. They do not supply the computed score or gradient.
Their original allocations stay at Runtime, as do gate identifiers, wire/rank
ownership and blocker/claim metadata. An initial preview extracted these
allocations; review rejected that scope before any formal activation.

The separate boundary-gate analytic expected value/gradient remains an
independent oracle. It must not call the implementation it validates.
Runtime also retains learning-rate checks and SGD updates; no optimizer
abstraction or generic array-provider layer is introduced.

## Compatibility and incoming integration

Existing Runtime functions and every existing Simulation definition retain
signatures, decorators, module/qualified names and behavior. No old path is
removed in this partial numerical extraction; each remaining module still owns
its execution lifecycle. No compatibility alias or forwarding is created.

Three incoming Runtime donor blobs are individually SHA-verified against the
complete non-truncated #1084 tree at
`f820620faf9f15d3c95a700c55a6adbf06ffafbc`. Backward is wholly byte-identical;
the other two Runtime files require preserving the parent series' existing
numerical extractions. The Simulation destination was created by that series
and is absent in the incoming tree. The later `667ccbaf...` and `831e551d...`
increments change no product path. A subsequent eleven-commit, thirty-one-file
increment to `2b7077f0235982609bbaed3ab1311e01347bde6b` is individually
blob-verified and likewise changes no product path or candidate binding.
A further six-commit, thirteen-test-file increment to
`4d6e479acaddf6414e5b05ca7016ab287da2fd30` is likewise hash-verified,
changes no product path and preserves all candidate bindings.
A subsequent six-commit, six-test-file increment to
`ceb270ab782b65abf75622565afec56cdae0b725` is also blob-verified and
changes no product path. The original donor comparisons remain bound to
the qualified parent Git blobs, independently of this active worktree.
The changed incoming Python source set has 3,275 files; the current 569-row
parent manifest still reports seventeen raw old imports, with no diagnostics
added, removed, filtered or normalized. Carry the parent-owned Simulation kernels
and new local responses together; do not restore the old inline numerical work.

#1084 remains Draft/conflicting. Its direct Construction/IR/compiler,
native/development distribution and root-tool/config integration remains gated.
Rust implementation remains deferred.

See the [acceptance record](runtime-mps-local-numerical-acceptance.md).
