# Static bounded loop synthesis delivery

Date: 2026-10-31. Unreleased development evidence. The human instructed merging
#978 and continuing the sixth plan. #978 merged as
8aec335f477272d8e5f33a75af6f56b9e0b1a502 after 57 successful checks and three
configured skips on 2a60f76f806014115b7d9065ff3669bcdf2b422e. This slice adds
the bounded classical loop part of stage 2. See the
[semantic contract](../api-changes/FQ-SYNTH-STATIC-LOOPS-20261031.md).
Independent trials remain 0/2 and release authorization remains false.

## User computation

Save the following in a source-retaining Python file, or run
python -m examples.synthesis_static_loop for both numeric and predicate loops:

```python
import flagquantum as fq
from flagquantum import synth


def weighted(x: int, y: int, steps: int) -> int:
    total = x
    for i in range(steps):
        term = y * i
        total = total + term
    return total


program = synth.synthesize(
    weighted,
    inputs={"x": synth.UInt(1), "y": synth.UInt(1)},
    constants={"steps": 3},
    input_state=synth.InputState.uniform(),
    max_qubits=24,
    memory_limit_bytes=256 * 1024**2,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y", "result"))))
print(program.report["resources"])
```

The sum is x+3*y. The four input/output tuples are (0,0,0), (0,1,3), (1,0,1),
(1,1,4), each with probability 1/4. Users specify unsigned quantum widths,
classical steps, known preparation and constraints. The compiler infers result
widths, produces fresh carried SSA values and manages gates, physical indices,
workspace cleanup and reading. No decorator or manual loop expansion is needed.

The second example starts with flag=x==y and ORs x==value for value in
range(1,steps). With steps=3 it marks x==y or x in {1,2}. All input probabilities
remain 1/16; ten amplitudes are negative. Complete complex-state acceptance
establishes that phase, not the probability table alone.

## Scope and reused implementation

Only the existing Compiler Python frontend changes production behavior.
for/name/range uses 1-3 nonnegative classical integer bounds and positive step.
Bounds must be INDEX and cannot hide quantum work in a constant-return helper.
Global/closure replacements of range are rejected without calling them.
Range arithmetic is checked before iteration; total expansion is limited to
256 iterations across helpers and sequential loops. Only one active loop is
supported, including helper bodies. Ordinary assignments are the loop body.
Parameters and induction index are read-only; preinitialized locals can be
updated once per iteration with fresh SSA. New locals and index are scoped;
old aliases preserve their original value. Outer single-binding rules remain.

Empty ranges keep carried values unchanged but still validate the body once in
an isolated environment; provisional SSA and value IDs are discarded. At capture
completion, unused classical constants are omitted without deleting quantum
operations. This prevents loop scaffolding from blocking direct-product phase
optimization or inflating a tiny physical binding's width. Empty, alias-only
and 256-iteration alias loops all preserve the two-input zero-workspace mask.
A prior test assumed original capture IDs equaled inlined lifetime IDs; it now
reads the returned ID from the authoritative inlined program. The source test
for rewriting an input in a loop remains a rejection, with the precise
assignment diagnostic rather than the former all-loop statement rejection.

The same flat HybridProgram, verifier/ranges/lifetimes, kernels, two schedules,
value XOR, exact phase prefix/adjoint, finite target selection and Program/run/
read remain authoritative. There is no loop IR/opcode, new type, public API,
report field, serializer or executor. Core, Services, Runtime and Simulation
are unchanged. Native Qmod grammar and phase/target restrictions are unchanged.
Nested/while loops, augmented assignments, branches, return/break/continue,
loop else, effects and data-dependent quantum iteration remain refused.

## Actual example costs

Complete uniform preparations and target passes, steps=3:

| Example / objective / schedule | Total qubits | Workspace | Operations | Depth | Planned state buffer |
| --- | ---: | ---: | ---: | ---: | ---: |
| weighted / width / recompute | 14 | 9 | 121 | 91 | 256 KiB |
| weighted / depth / retain | 19 | 14 | 95 | 72 | 8 MiB |
| matched phase / width / recompute | 16 | 12 | 947 | 762 | 1 MiB |
| matched phase / depth / retain | 18 | 14 | 153 | 122 | 4 MiB |

Costs are actual checked programs, separate from raw source costs. They are
finite candidate choices, not a global optimum, hardware speedup or process
peak-memory promise. An algebraically simplified manually expanded source can
produce a different gate sequence; correctness is compared on complete states.
The largest new external-reference test uses 20 qubits (16 MiB complex128).

## Acceptance and verification

44 new Compiler checks cover exact ranges/SSA/aliases, classical helpers/qfunc,
range forms and empty loops, scope, readonly bindings, shadowing, nested/helper
loops, hidden quantum bound work, iteration and expanded-operation budgets.
28 new real CPU/CLI checks include 128 basis executions of looped and explicitly
expanded sums under both schedules, covering all input pairs and all eight XOR
destination values, and 64 public basis executions for all 16 two-bit predicate
inputs in both modes/objectives. They also cover arbitrary coherent targets,
an external reference, full complex preparations, all zero through three
numeric loop counts, empty/no-op loops, zero-workspace phase preservation,
source/resource/memory refusals, plan identity and the public CLI. Tolerance is
1e-12 without discarding global phase. Final new suite: 72 passed.

The source coverage suite passed 156 checks. New/changed executable statements
72/72 and branch arcs 48/48 are covered; whole-frontend aggregate coverage is
94%, not repository coverage. Remaining misses belong to existing paths.
Strict mypy passes on 877 sources (the package plus this example). Ruff and
Black pass on all six changed Python files. Public API snapshots, architecture,
dependencies, vocabulary, repository language, capability maturity, generated
docs, local links and ownership classification pass. No public contracts or
snapshots change.

Separate regression: 1605 passed / 21 skipped, including the existing original-size
Qmod public file/text paths; public API candidates add 12 passed. Together with
the new 72 checks, these are 1689 distinct passing tests and 21 configured skips.
The local default run has 4664 passed / 121 skipped / 4842 deselected, then stops
at the same three pre-existing native QFT fusion assertions in
`tests/team/simulation/test_controlled_phase_graph.py` (44 operations versus
28 expected). This is not a green default run; no Simulation code or expectations
are changed. CI will be recorded on the submitted PR. Trials/release, all-corpus compatibility,
general quantum loops/control blocks and accelerator/gradient admission are
separate gates; no approvals or trial records are fabricated.
