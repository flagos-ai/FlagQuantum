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

## Gate preparation follow-up after PR #1345

The independent follow-up is based on the qualified PR #1345 source
`90be4e0cb14a44937f095f68b7cdd1997ad4122a`. Its CI qualification does not
qualify this follow-up's unpublished working tree.

| Existing Runtime calculation | Authoritative numerical implementation | Runtime responsibility retained |
| --- | --- | --- |
| `mps/lowering.py`, two successive-gate products | `simulation/numerics/jax.py::_jax_compose_gate_matrices` | Instruction order, pending-wire flushes and nearest-CX recognition |
| `mps/lowering.py`, local operator stack | `simulation/numerics/jax.py::_jax_stack_gate_matrices` | Layer grouping and fast-path admission |
| `mps/shards.py`, rank-four zero-site allocation | `simulation/mps/jax/batched.py::jax_mps_initial_batched_site` | Optional backend admission, rank ownership and device placement |

These are existing inline calculations, not whole-definition relocations. The
589-row relocation ledger is unchanged. No compatibility alias, forwarding
module, package or public contract is added. Existing private Runtime signatures,
parser array representations, dtype selection and multiplication precision stay
unchanged. The `parameters` argument of circuit lowering remains unused; consumed
angles come from instructions. No new public batch or differentiation capability
is claimed. Evidence-only RY/RXX allocations and independent analytic checks
described above remain at Runtime.

Before editing, 50 original outcomes were captured from an independently
installed native wheel: both lowering paths, full site arrays and complete
amplitudes, complex64/complex128, three multiplication precision settings,
rank ownership, empty batches and negative-size/channel refusals. The actual
working source matches all 50 outcomes byte for byte. Replacement tests call
unchanged Runtime consumers after replacing Simulation implementations; a
separate two-by-two operator test verifies JIT and parameter differentiation
without claiming differentiation through every MPS decomposition. Remaining
published-head CI qualification remains independently required.

The incremental #1084 comparison from `78c2cd1e...` to
`7f4708d6a8a15bd13c32a208f9e58cda8445498d` contains eight commits and
24 paths in compiler/register implementations, their tests and audit documents,
with no overlap with this follow-up. This is an incremental overlap check, not
a claim that #1084 is integrated. Its directly coupled migration remains gated.

The complete actual-source default tier passes **21,888 tests with 851
conditional skips**, in 479 seconds including startup. All 22,717 earlier
default cases remain, with exactly 22 new boundary tests; the one earlier
Git-provenance timeout skip now passes without changing that test. The real-JAX
seven-module source suite and its separate installed-artifact execution each
pass **194 tests with six unchanged device-count skips**. All 200 outcomes
match between source and installation. The installed run verifies that all
433 loaded product modules resolve to its own site-packages.

The new native wheel SHA-256 is
`3fcb4a6ab8270a095af29eaf9c0dcdd98d1561f1fa79de54b5237bf4df890f3f`.
Both native binaries are packaged, and all 1,265 product Python files match
source, wheel and installation. The installed artifact preserves all 50
original outcomes byte for byte. Complete source quality checks and fixed
hooks pass after correcting two local filesystem conditions: Ruff uses a
task-owned cache, and the EOF hook receives the file access required by its
`rb+` check. The initial environmental failures remain recorded. Installation
initially used a task interpreter without pip; the existing build interpreter's
pip installed the same wheel into the new isolated environment. No dependency
manifest, frozen contract or previous scientific test is modified.

All 24 incoming increment blobs are individually hash-verified. The unfiltered
changed-incoming source audit retains the same seventeen diagnoses under both
the 586-row parent and 589-row active relocation ledgers. One initial audit
bookkeeping command referenced the new report instead of the preserved prior
report; correcting that local path changed no diagnostic or source. These
checks do not close the final integrated G0 audit.
