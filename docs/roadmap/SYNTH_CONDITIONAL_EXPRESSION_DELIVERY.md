# Coherent conditional expression delivery

Date: 2026-10-31. Unreleased development evidence. The human instructed merging
#979 and continuing the sixth plan. Its head
95f357212b56c5dd184c925c914e78e8c67d2500 passed 57 checks with three configured
skips; merge is 94cfd28837b1a63b6938e0ecd2ab4fd682c7d2db. This slice extends
stage 2 with pure conditional value selection. See the
[semantic contract](../api-changes/FQ-SYNTH-CONDITIONAL-EXPRESSION-20261031.md).
Independent trials remain 0/2 and release authorization remains false.

## User workflow

Save the following in a source-retaining Python file, or run
`python -m examples.synthesis_conditional`:

```python
import flagquantum as fq
from flagquantum import synth


def smaller(x: int, y: int) -> int:
    return x if x < y else y


program = synth.synthesize(
    smaller,
    inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
    input_state=synth.InputState.uniform(),
    max_qubits=20,
    memory_limit_bytes=64 * 1024**2,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y", "result"))))
print(program.report["resources"])
```

Every `(x,y,min(x,y))` tuple has probability 1/16, and the result width is two
bits. Users provide unsigned widths, preparation and constraints, then ordinary
pure computation. They do not design a comparator, conditional copies, physical
maps, temporary storage, inverse gates or bit-string decoding. An ordinary
function is enough; qfunc remains an optional source snapshot.

The same example module marks `(x<y) if enable else (x==y)` in phase mode for
one-bit inputs. All eight input probabilities stay 1/8; exactly three amplitudes
are negative. Complete complex-state acceptance establishes this relative
phase, rather than the probability table alone.

## Semantics and architecture

The predicate must be exactly one quantum bit or classical integer 0/1;
comparison/Boolean expressions already produce admitted one-bit values. Both
pure branches are checked. An unselected invalid call, type or operator still
fails before planning. Classical guards fold to the selected expression;
already captured quantum branch work is retained and cleaned conservatively.
This is declarative capture, not host Python branch execution or measurement.
Nested selections, explicitly registered helpers, aliases, optional qfunc and
static loop assignments reuse the existing source rules and budgets.
Statement-level if/else, generic unitary control, mutable branch merges and
measurement feedback remain unsupported. Native Qmod syntax is unchanged.

One private `qint.select(predicate, yes, no)` operation extends the existing
HybridProgram/SSA. Unlike a sum of alternative products, its proven bound is
the branch maximum, so one-bit selections retain the one-bit phase contract.
The existing verifier rejects malformed guards, arity, attributes, regions and
insufficient widths. Lifetime analysis records every unique quantum dependency
needed by inverse computation. No new type, region, contract object, serializer,
identity or executor is needed.

The current Compiler arithmetic emitter zero-extends each branch and XORs
`no`, `predicate AND no`, and `predicate AND yes` into fresh result storage.
It uses X/CX/CCX with zero additional kernel scratch. Repeated predicate/source
controls simplify without flipping borrowed input qubits. Both retain and
recompute schedules retain their existing allocation, rematerialization and
exact cleanup. Phase mode exactly reverses the actual emitted prefix.

All production changes are inside the Compiler construction profile. Core,
Algorithms, Services, Runtime, Simulation, public signatures, snapshots and
contracts are unchanged. Existing target/resource selection, Program/run and
named reading remain authoritative. The old Boolean suite's rejection of
`x if y else 0` becomes an explicit positive one-bit selection check; all
bitwise rejections remain and real truth-table execution covers the migrated
case under both schedules.

## Measured actual costs

Complete uniform preparation and target passes, CPU complex128:

| Example / objective | Selected schedule | Total qubits | Workspace | Operations | Depth | State buffer |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| two-bit minimum / width | retain | 11 | 5 | 98 | 80 | 32 KiB |
| two-bit minimum / depth | retain | 11 | 5 | 98 | 80 | 32 KiB |
| one-bit conditional phase / width | retain | 8 | 5 | 68 | 60 | 4 KiB |
| one-bit conditional phase / depth | retain | 8 | 5 | 68 | 60 | 4 KiB |

Both existing schedules are evaluated. Both objectives select retain for these
examples; this is not a claim that the objectives always differ or retain is
optimal. Planned dense-state buffer is not process peak memory or hardware
performance. The largest new actual external-reference state has 12 qubits
(64 KiB complex128).

## Acceptance

32 new Compiler checks cover exact maximum bounds and one-bit preservation,
source locations, constant folding, retained unselected quantum work, helper/
qfunc/alias/loop composition, unselected errors, guard truthiness, forged SSA,
statement-branch refusal, nesting budget and the existing 256-bit cap.
41 real CPU/CLI checks cover 368 basis executions over nine selection forms,
both schedules and every XOR destination. These include unequal widths,
constants, nested choices, repeated aliases and the formerly rejected Boolean
mask. Independent complex references cover numerical minimum with every target,
one-bit value/phase, an entangled external reference and workspace zero without
discarding global phase. Tolerance is 1e-12. Public checks cover all 16 minimum
input pairs in uniform superposition, a nonzero basis preparation, complex
phase preparation, both objectives, line target conversion, helper/qfunc/loop
composition, resource/memory failures, no planning after source rejection and
the complete example CLI. Final new suite: 73 passed.

The new 73-test suite and 454 existing Compiler tests pass in coverage runs
(527 checks, overlapping the regression suite). All 32/32 added executable
statements and 24/24 added branch arcs are covered. The four touched production
modules aggregate 96% coverage; frontend/lifetimes/lowering/profile aggregates
are 95%/98%/97%/97%, not repository-wide coverage. Remaining misses belong to
existing paths. Strict mypy passes on 877 sources (the package plus the example).
Ruff, current and minimum Black pass on all eight changed Python files. Public
API snapshots, architecture, dependency/vocabulary/language/maturity checks,
generated docs, 762 Markdown links and ownership classification pass. The
separate public API candidate suite passes 12 checks.

The local default run has 4698 passed / 121 skipped / 4883 deselected, then
stops at the same three pre-existing native QFT fusion assertions in
`tests/team/simulation/test_controlled_phase_graph.py` (44 operations versus
28 expected). This is not a green default run. No Simulation implementation or
expectations are changed.

Separate broader regression: 1677 passed / 21 skipped, including six
original-size file/text executions of the three unmodified Qmods. With the new
73 checks and 12 public API candidates, distinct selected acceptance totals
1762 passed / 21 skipped. CI will be recorded on the submitted PR. This slice does not promote trial/release status, the original
three-Qmod count, general control-flow admission or compatibility with all 216
Classiq files. Accelerator and gradient admission retain their existing scope.
