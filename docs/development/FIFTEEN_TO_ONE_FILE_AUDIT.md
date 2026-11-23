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
