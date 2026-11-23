# Complete 15-to-1 factory source closure

The complete frozen source is Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Parent | Source file | Frozen blob | Native owner |
| --- | --- | --- | --- |
| Q-0910 | surface_code/fifteen_to_one_factory.py | d4c36592c0ced4268ac11791991b34ec7a566c7a | qec/resources/fifteen_to_one.py |
| Q-0911 | surface_code/fifteen_to_one_factory_test.py | 616820dc041eeeb3dc5452de7dbb30dfd735ad55 | tests/unit/test_fifteen_to_one_factory.py and test_fifteen_to_one_file.py |

## Definition and dependency closure

`FifteenToOne` maps to `FifteenToOneFactory`. Its three distances map to
`x_distance`, `z_distance` and `measurement_distance`. It now implements the
already accepted `MagicStateFactory` interface. Footprint, expected production
cycles, cumulative error, `_final_state`, `p_fail` and `p_out` all have direct
native counterparts. The complete source `_build_factory` schedule is owned by
`_factory_ir`; the existing `to_ir` entry binds its supplied scaler and physical
error to the shared logical-error model. New bound-model methods also honor
nondefault thresholds. The existing concrete `estimate` path shares the same
state and postselection implementation.

State and postselection use separate bounded eight-entry caches, shared by all
factory queries. Source p_fail/p_out caches are consolidated into the shared
postselection cache. A counted executor verifies one 100-operation execution
across all probability, cycles, error and estimate queries. No new circuit or
simulation implementation is introduced. The native density-matrix executor,
Pauli-channel emitter and shared gate-budget conversion remain authoritative.
A real Beverland consumer accepts a source-backed density implementation through
the same native factory boundary.

## Reused and additional source evidence

The preceding QEC06 evidence in `test_fifteen_to_one_factory.py` checks all 100
operations, operands and nonzero channel matrices against the unmodified
source, with complete complex density matrices for four paper examples and the
Beverland case. It covers every source rotation/error slot and the complete
postselection projector. Channel matrices use absolute tolerance 1e-15 and full
densities 2e-15. Rejection and output probabilities use 2e-14, and cumulative
errors use 4e-12. These numerical comparisons do not claim bitwise equality.

The whole-file extension checks the frozen source/test hashes, all interface
methods for four distance triples and two threshold/scaler pairs, empty and
mixed demand, symbolic cumulative error, original symbolic cycle TypeError,
parallel composition and bounded reuse. Source tests run unchanged, including
all four paper cases and invalid distances.

The four postselection checks are quantum channels projected onto X=+1. This
model returns an unmeasured density matrix and conditional scalar estimates.
It adds no classical measurement histories or clean ancilla claims. Emitted
100-operation channel counts, logical gate demand and modeled physical resource
costs are recorded separately.

## Native boundaries

Existing native positive-integer distance validation raises ValueError while
the source uses AssertionError. Existing `to_ir`/`estimate` validation restricts
physical error and scaler. The shared model methods admit the bound scheme and
retain the original arithmetic for supported channels. Symbolic production
cycles remain unsupported. Zero-acceptance postselection retains the existing
native explicit failure. Primitive/channel evidence and physical assumptions
remain independently owned.

All 267 concentrated resource regressions and five unchanged original source
tests pass. Both changed Python files pass strict mypy, Black and Ruff.
Core-team scope, architecture, IR and API checks pass. Q-0910 and Q-0911 close
with the explicit native validation and numerical boundaries above.


## Reopened high-demand integer boundary and optional collection

On 2026-11-19 the complete dashboard consumer exposed an untested ceiling
boundary in the preceding acceptance. For physical error 1e-4, Gidney-Fowler
scheme, distances (7, 3, 3) and T-equivalent demand 2,500,000,000,091, source
production cycles are 45,137,839,388,431 and native cycles are
45,137,839,388,432. Rejection probabilities differ by 1.5543122344752192e-15.
The complete density matrices remain within the previously recorded tolerance,
but this does not establish integer-cycle equivalence at large demand.
Q-0910/Q-0911 are reopened. Earlier gate, density and bounded-demand evidence is
retained, and the exact dashboard regression remains failing. See
`PHYSICAL_RESOURCE_DASHBOARD_AUDIT.md` in the current working batch and the
persistent `g2-dashboard-cycle-witness.log` for the complete witness. This
supersedes the earlier unconditional whole-file closure statement.

The independent optional-collection repair guards NumPy and source packages in
all three distillation test modules. Explicit channel-payload assertions also
establish the matrix invariant used by strict typing. A missing-NumPy import
probe confirms that each module reaches its dependency skip, while the complete
qualification environment executes every case. All 451 concentrated resource
regressions pass without skips, four changed test modules pass strict mypy,
Black and Ruff, and architecture, IR and protected API checks pass. This repairs
collection, without resolving the high-demand precision boundary. The four-file
batch has zero dependency-policy findings. The whole repository retains 706
findings, down from 711, recorded in the persistent
`g2-distillation-collection-dependencies.log` and dependency-scope JSON.

## Shared numerical operation-order isolation on 2026-11-23

A fresh immutable-source intake reproduces the complete high-demand witness.
The factory, test, channel and QEC-scheme source Git blobs are recorded in
`g2-factory-numerical-intake-20261123.json`. Earlier full schedule and paper
receipts are reused. The source simulator first produces five independent H
states and merges their density products. Their shared initial matrix entry is
0.031250000000000035. The native expanded-matrix H route has the previously
recorded distinct rounding.

Using that identical source initial matrix and the identical original Kraus
matrices isolates the remaining execution arithmetic. Native full BMM,
axis-based Torch einsum, and explicit one-qubit slice application all yield
final density differences of 3.0117855110836314e-17 from the original simulator.
The first rotation already differs by 3.469446951953614e-18. Merely selecting an
axis contraction or one-qubit slice implementation does not close the witness.
Direct original apply_channel replay from the same initial matrix is bitwise
identical to the original final density.

The original channel protocol converts mixtures into Kraus matrices before
execution. The prior hypothesis that retaining mixture probabilities alone
would reproduce this path is rejected. All 95 non-H source operators already
have bitwise identical nonzero native Kraus payloads. Adding a mixture payload
would not address the observed contraction difference.

Final projection is a further independent boundary. On the identical original
density, NumPy computes acceptance 0.9969462564300695 and Torch computes
0.9969462564300696. The source-H/native-BMM experiment computes NumPy acceptance
0.9969462564300697. At demand 10^14 the resulting cycle values are respectively
1,805,513,575,471,519 and 1,805,513,575,471,518. The baseline native route still
returns 1,805,513,575,471,522 at that demand.

Q-0910/Q-0911 remain open. Exact full-range cycle parity requires resolving
product-state initialization, complex contraction arithmetic and postselection
reduction together in their shared numerical owners. The supported Torch
execution/device/gradient contract must remain explicit during that work.
The current production executor and source references retain their accepted
behavior. The experiment supplies no new channel vocabulary, execution policy
or accepted integer-range limit. Remaining independent G2 files can proceed.

## Source-ordered sparse Kraus contraction on 2026-11-23

Further isolation fixes the initial density and all source Kraus matrices, then
compares every channel step. For a matrix with at most one nonzero coefficient
per row, the channel can gather its selected input entries directly. The source
contracts the conjugated matrix into the column axes with its coefficient as
the first complex-multiplication operand. Moving that coefficient to the other
side changes floating-point rounding even though multiplication is algebraically
commutative. Separate real/imaginary products and fused multiply-add variants
fail the same witness. Merely transposing the dense right BMM also fails.

The existing Simulation apply_kraus_density now owns this matrix-first sparse
contraction for finite, constant CPU operators. Rows may share a selected column
and may be entirely zero. Batched matrices and state gradients are supported.
Trainable operator entries retain the existing dense product, including
derivatives at currently zero entries. Nonfinite tensors and other devices
retain their existing dense product and catalog ownership. No public entry,
channel vocabulary, source object or reference matrix is changed.

The complete 95-channel suffix now agrees bitwise with unchanged source
apply_channel at every step for distance triples (7,3,3), (11,5,5) and (17,7,7),
using both the source-prepared density and an independent coherent complex
input. This is local CPU complex128 evidence. Complex64/complex128 state
and batched-operator derivatives also agree with independent dense products.
Existing dense catalog routing remains covered separately. The normal factory
and channel consumers use the repaired owner through their existing IR path.

The complete factory remains open. With native initial preparation, demand
2,500,000,000,091 still returns 45,137,839,388,432 versus source
45,137,839,388,431. At demand 10^14, native now returns
1,805,513,575,471,521 versus source 1,805,513,575,471,519. Native rejection is
0.0030537435699318216 versus source 0.0030537435699304893. Independent H-factor
preparation and final projection remain unresolved. This replaces the earlier
contraction arithmetic hypothesis with a verified source-ordered channel owner.
It does not establish complete-factory integer-cycle parity.

Channel instruction counts, operands, matrix payloads, logical demand, workspace
and measurement semantics receive zero additions. The 100-operation schedule
retains five H instructions and 95 channels. Physical modeled resources remain
separate from numerical execution arithmetic. Persistent receipts use
ftqc-reference-env/g2-density-source-contraction-*20261123 and the corresponding
factory arithmetic probes. Earlier immutable source and complete test-file
hash verification is reused.

Validation passed 31 focused checks, 242 concentrated density/channel/resource
checks, 145 relevant local integration checks and strict typing for both Python
files. Two concentrated accelerator checks skip because CUDA is unavailable.
The local integration run reports 18 existing wires-deprecation warnings.
Black, Ruff, Simulation scope, language, architecture, IR and API checks pass;
the sorted complete 720-finding dependency baseline is identical.

An initial broad integration run was interrupted after 117 seconds with
465 passes, 109 failures, 13 errors and 226 skips. It exposes unavailable
notebook socket permissions, localhost distributed setup and a LaTeX iswap
rendering gap. The published pre-change density module independently reproduces
both a representative notebook socket failure and the iswap rendering failure.
The broad suite is not certified by this batch. Its full diagnostic log and
baseline replay receipts are retained. The bounded 145-case affected integration
selection passes separately.

## Complete source arithmetic closure on 2026-11-23

The initial-state and projector boundaries above are resolved by shared native
Simulation owners. The earlier reopened and intermediate channel-only outcomes
remain historical evidence of the diagnosed failures.

### Independent density factors

The existing density executor now retains a leading sequence of independent
one-qubit operations as separate density factors, then combines them in wire
order before executing the remaining IR. Each instruction still executes once.
Cirq's H subtraction/scaling sequence is ported into the density-axis owner with
its source notice. It produces the original one-qubit entry
0.5000000000000001 and ordered five-factor entry 0.031250000000000035.

The optimization admits known CPU zero inputs without a state gradient and the
existing complex64/complex128 precisions. Circuit and IR entry points share the
same eligibility check. Entangled input and live input gradients retain their
existing execution path. Explicit matrix overrides and batched gate parameters
keep their actual values and derivatives. This fixes a regression caught during
the batch where the two entry points initially chose different zero-state paths.
Statevector product components remain with their existing pure-state owner;
this mixed-state prefix reuses the existing density executor and matrix builder.

### Complex trace reduction

The reference environment uses NumPy 2.5.4 and Cirq 1.7.0 with unchanged locked
Qualtran 0.7.0 source. The original NumPy complex pairwise reduction declaration
was fetched read-only and retained as numpy-2.5.4-loops_utils.h.src beside the
worktree. Simulation density_reduction.py ports its four-complex-accumulator
ordering, 64-complex-entry blocks, balanced recursion and tail reduction into
Torch operations. NumPy's complete BSD-3-Clause notice accompanies that owner.
The trace remains differentiable and supports batched, strided diagonals.

A two-way summation candidate fitted only 10 of 24 initial source cases and was
rejected. With the correct order, all 24 original projection matrix products
and accepts agree exactly. Thirty-two independent complex64/complex128 tests
cover empty, short, block-edge, tail and recursive trace sizes through 129, with
distinct binary scales and complete gradient checks against the unchanged
reference and independent analytical gradients.

### Factory composition

The existing QEC factory consumes the shared trace. Reciprocal postselection
scaling and the target row's complex phase follow the source arithmetic before
forming its outer product. All production execution remains native Torch and
scalar Python math. No Qualtran, Cirq or NumPy runtime is called by the factory
or its new Simulation owners. Existing bounded caches, physical assumptions,
validation differences and symbolic restrictions retain their documented owners.

The complete 72-case grid covers six distance triples, physical errors
0/1e-4/1e-3, scalers 0.03/0.1 and thresholds 0.01/0.02. It compares every complex
entry of the final density bitwise, and compares rejection and conditional
output error exactly. Seven demand values include zero, the original
2,500,000,000,091 witness, 10^14, 2^53+1 and 10^18. Integer-cycle comparisons
convert the source float to an integer first, preventing large integer
comparisons from silently rounding through float. All 504 demand cases and
cumulative factory errors agree exactly.

The repaired original witness returns 45,137,839,388,431 on both sides. At
10^14, both return 1,805,513,575,471,519. Rejection on both sides is
0.0030537435699304893 and the complete final density difference is zero.
These are measurements in the qualified CPU complex128 environment. Source
mathematical validation and unrelated backend qualifications remain separate.

The original full source/test definition map and five original unmodified tests
are reused. All 100 emitted operations, operand ordering and Kraus payloads
retain their previous evidence. Actual numerical H/channel operations add no
quantum instructions, measurements or work. The four X=+1 checks are still
unmeasured projector postselection, with no measurement-history or clean-ancilla
claim. Physical modeled costs remain separately reported.

### Acceptance

Q0910 and Q0911 close for the complete factory source and its original test
file. The concentrated dependency regression passes 373 cases, including all
five unchanged original tests. Two CUDA cases skip because the accelerator is
unavailable. The downstream resource consumers pass 146 cases and the affected
local integration selection passes 145 cases with 18 existing warnings.
Strict typing passes all six changed Python files. Black, Ruff, owning-team
scope, language, architecture, IR and protected API checks pass. The complete
720-finding dependency baseline is identical. The earlier broad-suite diagnostic
remains open as documented above. CPU arithmetic qualification and the original
source's mathematical and backend boundaries remain explicit.
