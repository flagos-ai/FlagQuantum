# Unsigned comparison and chain delivery

Date: 2026-10-31. Unreleased development evidence. The human instructed merging
#981 and continuing the sixth plan. Its exact head
30c68501f1b070ff13bde4a20332284ed2175237 passed 57 checks with three configured
skips and merged as 7639db0c477bbb0a9350aa22cf7310276f0797e4. This stage-2
slice completes unsigned comparisons and chained interval predicates. See the
[contract](../api-changes/FQ-SYNTH-COMPARISON-CHAINS-20261031.md).
Independent trials remain 0/2 and release authorization remains false.

## User computation

Save the following in a source-retaining file, or run
`python -m examples.synthesis_interval` for both value and phase:

```python
import flagquantum as fq
from flagquantum import synth


def allowed(x: int, low: int, high: int, excluded: int) -> bool:
    return low <= x <= high and x != excluded


program = synth.synthesize(
    allowed,
    inputs={"x": synth.UInt(3)},
    constants={"low": 1, "high": 6, "excluded": 3},
    input_state=synth.InputState.uniform(),
    max_qubits=24,
    memory_limit_bytes=256 * 1024**2,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "result"))))
print(program.report["resources"])
```

This accepts 1, 2, 4, 5 and 6. All eight input/result pairs have probability
1/8. In phase mode, precisely these five basis amplitudes gain -1 and all input
probabilities stay 1/8. The complete complex state, not the probability table,
establishes the phase effect.

The user supplies arithmetic logic, unsigned quantum widths, explicit classical
bounds, preparation and constraints. They no longer rewrite <=/>=/!= in terms
of < or == and not, split interval checks, design comparator/control gates,
manage workspace or decode bits. The entry, Program and named reader are the
same; qfunc is optional.

## Semantics and reuse

Only the Compiler Python frontend changes production behavior. Six relations
normalize to existing equal/less and one-bit negation. Chains conjoin adjacent
pairs using the same existing emitter as Boolean and/or composition. Each
operand captures once, in source order. Swapping a pair for > or <= never
changes which original operand feeds the next pair. All parts validate after
a false prefix; no user code runs and no input is measured or sampled.
All-classical pairs and conjunctions fold exactly. Mixed widths zero-extend;
results stay one bit and compose with helpers, aliases, qfunc, conditional values
and static loop assignments. Existing safety budgets remain unchanged.

Identity/membership, bool literals/parameters, signed/fixed-point values,
implicit multi-bit truthiness and statement-level control remain unsupported.
The five previously refused comparison forms now have explicit positive
acceptance. The old Boolean false-prefix negative uses an unsupported identity
operator instead; the formerly rejected valid chain has an added positive test
and real full-domain execution. Unsupported source still cannot hide behind
classical falsehood.

HybridProgram/SSA, verifier/ranges/lifetimes, arithmetic kernels, retain/recompute,
value XOR, exact phase-prefix inverse, target/resource selection, Program/run and
reading all remain authoritative. No new opcode, type, region, serializer,
executor, public signature/schema/snapshot/contract or dependency is added.
Core, Algorithms, Services, Runtime, Simulation and native Qmod grammar are
unchanged by this slice. Main already contains the independent MPS bounded
probability work; it is retained without modification.

## Actual complete example resources

Uniform preparation and target passes, CPU complex128:

| Mode / objective | Schedule | Total | Used | Workspace | Operations | Depth | State buffer |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| value / width | recompute | 13 | 12 | 9 | 234 | 191 | 128 KiB |
| value / depth | retain | 15 | 14 | 11 | 120 | 83 | 512 KiB |
| phase / width | recompute | 12 | 11 | 9 | 256 | 207 | 64 KiB |
| phase / depth | retain | 14 | 13 | 11 | 112 | 77 | 256 KiB |

Total slots include an allocation left unused by subsequent gate optimization;
used is the actual instruction footprint. Workspace is the source schedule's
peak allocation, not a claim that every slot is used in the final instructions.
State buffer follows total slots and is distinct from process peak memory.
These are checked finite candidates, not globally optimal resources or hardware
performance. New external-reference states have at most 14 qubits / 256 KiB
complex128. Public tests retain explicit 24-qubit / 256 MiB ceilings.

## Acceptance

51 new Compiler checks cover all relations, exact one-bit ranges/source
locations, migrated positive cases, classical boundary truth, mixed-direction
chains, shared helper capture, qfunc snapshots, false-prefix refusals and the
expanded-operation safety budget. 83 real CPU/CLI checks include 736 basis
executions over all six relations and eight chain forms, mixed widths,
constants on both sides, aliases and every XOR target under both schedules.
There are 32 public basis executions for all eight interval inputs in both
modes/objectives. Full complex states cover coherent XOR destinations, absolute
phase, an external reference and workspace zero; tolerance is 1e-12 without
removing global phase. Raw numerical probes explicitly request complex128.
Eight additional direct public alias probes (x>=x, x<=x, x!=x and x>x under
both objectives, UInt(2), uniform phase preparation) preserve the exact +1/-1
global phase on every input; these are supplementary probes, not added pytest
counts. Public complex preparation, line target conversion, helpers/qfunc/loops/
conditional values, source-before-plan rejection, budgets/memory and the
complete example CLI also pass. New suite: 134 passed.

Coverage runs add 481 existing Compiler checks to the new 134, for 615
passing checks overlapping the broader regression. All 28/28 added executable
statements and 12/12 added branch arcs are covered; the full frontend aggregate
is 95%, not repository coverage. Remaining misses belong to existing paths.
Strict mypy passes on 878 sources (the package plus the public example).
Separate public API candidates pass 12 checks. Ruff/current and minimum Black pass on all six changed Python files. Public
API snapshots, architecture/dependency/vocabulary/language/maturity checks,
generated docs, 764 Markdown links and ownership classification pass.

The local default run has 4744 passed / 121 skipped / 4966 deselected, then
stops at the same three pre-existing native QFT fusion assertions in
`tests/team/simulation/test_controlled_phase_graph.py` (44 operations versus
28 expected). This is not a green default run; no Simulation implementation or
expectations change.

Separate broader regression: 1745 passed / 21 skipped, including all six
original-size file/text executions of the three unchanged Qmods. Together with
the new 134 and separate 12 public API checks, distinct selected acceptance
totals 1891 passed / 21 skipped. CI will be recorded on the submitted PR. The original three-Qmod count, independent trial/release gates,
general quantum control/within-apply, all-216 compatibility and accelerator/
gradient admission retain their separate requirements.
