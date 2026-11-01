# Ordered phase borrow delivery

Date: 2026-11-01. Unreleased development evidence. The human instructed merging
#982 and continuing the sixth plan. Exact head
cc6d69bad7d4617c7473b203580db424add3d203 passed 57 checks with three configured
skips; merge is 8577d6e3967bbd017ff57e0063915cfa5ffbe90e. This stage-2
slice establishes a closed basis-constant call and internal scratch scope before
general quantum function roles and within/apply. See the
[semantic contract](../api-changes/FQ-SYNTH-REGISTER-PHASE-20261101.md).
Independent trials remain 0/2 and release authorization remains false.

## User intent and contributor path

The existing public high-level path remains:

```python
import flagquantum as fq
from flagquantum import synth


def marked(x: int, y: int) -> bool:
    return x * x + y < 2


program = synth.synthesize(
    marked,
    inputs={"x": synth.UInt(1), "y": synth.UInt(1)},
    input_state=synth.InputState.uniform(),
    output_mode="phase",
    max_qubits=24,
    memory_limit_bytes=256 * 1024**2,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y"))))
```

Users supply the predicate, quantum widths, preparation and constraints. The
compiler owns the gates and inverse workspace cleanup. All four input tuples
have probability 1/4; three gain a minus sign. That sign needs complex-state
verification and cannot be inferred from the probability table.

This PR makes the same kind of predicate an ordered action on live registers
inside the existing register program. A later frontend can admit the action
without requiring a caller-owned flag, manual uncomputation or another executor.
There is no new public builder, decorator or role syntax in this slice.

Run `python -m examples.synthesis_register_phase` for the **private contributor
trial**. Its owned-IR producer declares four outputs, prepares x/y, copies a
basis reference from x, calls the predicate phase, flips x, computes a product,
calls the predicate again on product/y, then applies H to y for interference.
The private existing Services bridge returns the same checked Program; fq.run,
actual target mappings and named reading are reused. The reference lies outside
each phase call's argument set. Copying a basis encoding entangles it with x;
it does not clone an arbitrary quantum state.

The five nonzero logical output amplitudes (x,y,product,reference) are:

| Tuple | Complex amplitude |
| --- | --- |
| (0,1,0,1) | i/sqrt(2) |
| (1,0,0,0) | 1/sqrt(8) |
| (1,1,0,0) | 1/sqrt(8) |
| (1,0,1,0) | -1/sqrt(8) |
| (1,1,1,0) | 1/sqrt(8) |

An independent sum over the two prepared input bits derives these amplitudes,
including constructive/destructive interference, instead of replaying compiled
gates or discarding global phase.

## Roles, ownership and implementation

New private `qreg.phase(effect,*registers)->next_effect` has exactly one callee
attribute and no regions. The callee uses the existing closed unsigned function
profile and must return exactly one bit. Borrowed references must already be
initialized; their basis values and availability remain after the call. Their
phase may change. This is a derived basis-constant borrow, not a classical
constant, unchanged quantum state, consuming Input or fresh Output assignment.
Extra user trust/Const claims are rejected. Fresh initialization remains the
existing allocate/initialize/compute responsibility.

The same call checker serves compute and phase. Existing unique-actual wrappers
preserve repeated aliases without duplicated physical entry bindings or copied
states. The same arithmetic scheduler emits its selected prefix, marker and
exact prefix inverse, so every temporary is zero before the next ordered
operation can mutate a dependency. Each later call starts from that same zero
tail; all named output registers remain live. No mutable body can interleave
inside a phase call because its callee is the closed immutable expression model.

| Responsibility | Reuse or change |
| --- | --- |
| Quantum register/uint/effect records and semantic identities | Reuse existing HybridProgram/SSA; no type/schema/identity added |
| Fresh initialization and linear effect order | Reuse register verifier; add phase-call signature and one-bit borrow check |
| Alias handling and arithmetic call dispatch | Extend existing wrapper/emitter to phase mode with no value-result port |
| Predicate kernels, lifetime schedules and exact adjoints | Reuse existing arithmetic/phase implementation |
| Candidate/target checks | Share existing phase admission guard; preserve witness audits |
| Execution, layout-aware reading and memory checks | Reuse existing Services bridge, Program and Runtime |
| Public Python/Qmod grammar, general roles/within-apply | Unchanged; separate future admission |

All production changes stay in three Compiler construction files. Core,
Algorithms, Ecosystem, Services, Runtime, Simulation, public signatures,
snapshots/contracts, serialization and dependencies are unchanged.
There is no new contract record, executor, registry or general user control DSL.
General consuming Input, Output reference transfer, free/drop and arbitrary
within/apply effect validation remain unimplemented by this slice.

## Measured resources and target boundary

Complete zero-entry program including preparation, both phase actions, product
and copied reference, plus final interference:

| Objective / schedule | Total | Used | Peak workspace | Operations | Depth | State buffer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| width / recompute | 12 | 10 | 8 | 88 | 78 | 64 KiB |
| depth / retain | 13 | 11 | 9 | 58 | 50 | 128 KiB |

The four segment workspace peaks are (0,8,2,8) or (0,9,2,9); the combined peak
is 8 or 9, not their sums 18 or 20. Real full-state checks establish that the
shared internal tail is zero. Total retains allocation slots left unused by
later optimization; used is the final instruction footprint and workspace is
the source scheduling peak. These are finite measured candidates, not global
optima, process peak-memory or hardware-performance claims. Explicit test caps
are 24 qubits / 256 MiB; separate borrowed-segment proofs add one external
reference and require zero only on their internal workspace.

Native logical and PHASE_BASIS routing with final physical layout retained pass
full complex-state checks under both objectives. For this interleaving fixture,
restored routing is refused before planning: the existing normalizer cancels
restore SWAPs and the target's history audit no longer matches. The recompute
probe loses 104 restore witnesses. This preserves the already recorded
routed-SWAP witness limitation in the predicate-phase capability matrix. No audit is weakened and no unverified
mapping reaches Runtime. General routing-evidence repair is outside this role
slice; restored routing is not claimed for this fixture. Existing arithmetic
phase-target rules are otherwise unchanged.

## Acceptance

21 new Compiler checks and 25 CPU/CLI integration checks pass: 46 total.
They cover initialization/one-bit/type/callee/effect/attribute/region rules,
repeated actual borrows, no consumption/implicit initialization, exact target
admission, finite search and whole-program operation limits. All eight small
basis inputs execute for each schedule (16 runs) through a borrowed phase
segment. Arbitrary complex inputs entangled with an extra external reference,
true/false/data-dependent alias phases, and both schedules preserve absolute
phase and return workspace to zero at tolerance 1e-12.

Whole checked programs verify the independent interference state, reference
entanglement, mutations between calls, phase/compute workspace reuse, both
objectives, native/logical and retained-layout routing, named probabilities and
plan identity. Compiled all-true and all-false alias phases retain their exact
global sign under both objectives. A direct repeated-input product phase fits
one qubit with zero workspace and interferes to |1>. Invalid reads, projective
or unproven targets, resource/memory failures refuse before planning/execution.
The complete private example CLI runs. Raw segment proofs use arbitrary
initialized borrowed states; whole fresh-register programs still require zero
entry storage, and the two preconditions are not conflated.

Coverage runs pass 46 new checks plus 532 existing Compiler checks (578 total,
overlapping regression). All 34/34 added executable statements and 26/26 added
branch arcs are covered. Register verifier and lowering each have 100% coverage;
selection has 99%, and the three-module aggregate is 99%. This is measured
module coverage, not repository coverage. Strict mypy passes on 878 sources
(the package plus the contributor example). Separate public API candidates pass
12 checks. Ruff and current/minimum Black pass on all six changed Python files. API,
architecture/dependency/vocabulary/language/maturity/generated-doc checks,
766 Markdown links and ownership classification pass.

Separate regression: 1879 passed / 21 skipped, including six original-size
file/text executions of the three unchanged Qmods. With 46 new and 12 public
API checks, distinct selected acceptance totals 1937 passed / 21 skipped.
The local default run has 4765 passed / 121 skipped, then stops at the same
three pre-existing native QFT fusion assertions in
`tests/team/simulation/test_controlled_phase_graph.py` (44 operations versus
28 expected). This is not a green default run; no Simulation code or
expectations change. CI will be recorded on the submitted PR.
The 3/3 original-Qmod milestone, all-216 compatibility, independent usability
trials, public role admission and release/accelerator/gradient gates remain
separate requirements.
