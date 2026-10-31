# Coherent predicate synthesis delivery

Date: 2026-10-31. Unreleased development evidence. Builds on #971, merged as
2560f074dd59bfc1390f7b8b0c58d5109c118747. The human user approved the concrete
next step: single Python == and < predicates, shared lowering and full acceptance.
See the [narrow contract](../api-changes/FQ-SYNTH-COMPARISONS-20261031.md).
The [existing Program journey](SYNTH_PROGRAM_RUN_DELIVERY.md) remains the base.
Independent developer trials remain 0/2 and release authorization remains false.

## User experience

Run python -m examples.synthesis_predicate. The ordinary function
predicate(x, y) returns x*x+y<5 with optional bool annotation. The user supplies
UInt widths and named basis/uniform preparation, then calls synth.synthesize,
fq.run(program) and program.probabilities by logical name. There is no required
qfunc, gate sequence, physical index, workspace choice or bit decoding.
The result flag is displayed as an integer 0/1 in a logical tuple. Uniform input
remains coherent; the predicate neither measures nor runs a classical branch.

== and < use complete unsigned values with zero extension. Constants can occur
on either side; exact both-constant expressions fold without running user Python.
Quantum input is not folded from its prepared basis value. Repeated or aliased
operands retain their numeric identity. Optional bool return annotations require
one-bit results, including helpers; input annotations remain optional int.
Flags reuse the existing one-bit numeric encoding in arithmetic.

## Reuse and new implementation

Compiler adds qint.equal/qint.less to the existing closed profile and range
analysis, and recognizes ast.Compare in the existing source-retaining frontend.
It reuses SSA, source locations, physical binding checks, dependencies,
retain/recompute scheduling, candidate selection, target transforms and costs.
Both schedules use the same z ^= predicate(inputs) lowering for either target
bit, immutable input borrows and clean workspace.

The prefix-equality comparator from Algorithms is extracted into the existing
pure Compiler arithmetic-kernel boundary, also shared by the adder. Algorithms
keeps its public signature, validation and greater-than behavior and consumes
this same instruction kernel. Constant/padded bits need no allocated constant
register; aliases are handled before controls are emitted. Each comparison
locally restores its prefix flags and scratch. Compiler gains no Algorithms,
Circuit, Runtime, backend or numerical execution dependency.
Services, Program, run/read, public signatures/exports, schemas and candidate API
snapshots are unchanged. Qmod parsing/context and its three-original counts are
unchanged. There is no additional executor or truth-table enumeration.

## Measured resource costs

Raw lowering for two-bit x/y and a one-bit XOR destination, before target
transformation or input preparation:

| Expression | Schedule | Total qubits | Workspace | Operations | Depth |
| --- | --- | ---: | ---: | ---: | ---: |
| x*x+y<5 | retain | 22 | 17 | 207 | 140 |
| x*x+y<5 | recompute | 19 | 14 | 295 | 188 |
| x*x+y==5 | retain | 22 | 17 | 197 | 136 |
| x*x+y==5 | recompute | 19 | 14 | 285 | 184 |
| x==y | either | 10 | 5 | 63 | 56 |

The complete public uniform less-than example selects recompute by width and
actually executes 19 qubits, 14 workspace qubits, 273 operations and depth 178.
Those actual counts include four Hadamards and the target pipeline's
optimization. program.report.resources is verified against the actual bound
execution plan and returned result; source_resources retains the complete
pre-transformation cost. Raw and actual counts are not interchangeable.
No timing, optimality or hardware advantage follows from these costs.

## Acceptance and boundaries

Real CPU tests execute all 16 two-bit input pairs for x*x+y<5 and x==y, each
with initial XOR destination 0 and 1 under both schedules. Full complex128
reference states for x*x+y<5 and x*x+y==5 cover all inputs and both destination
values in a phase-bearing superposition, with tolerance 1e-12 and all workspace
returned to zero. This is 128 individual basis executions plus coherent tests,
not one randomly chosen input or a probability-only equivalence check.
The largest per-state buffer is 64 MiB at 22 qubits.

Additional tests cover logical/line targets, unequal widths, constants in both
positions, equal/partially aliased operands, range boundaries, helper and qfunc
composition, annotation errors, malformed private records and early syntax/
resource refusals. Independent exhaustive kernel permutations cover eq/lt/gt
and cleanup. Existing Algorithms comparator tests run unchanged.
The complete public example runs as a real subprocess and checks named output
against an independent 16-case reference.

Not admitted: chained comparisons, other relational operators, and/or/not,
bool parameters or literals, Python if/loops, native Qmod comparisons, general
quantum control, invert/within-apply, arrays, signed/fixed-point types or release.
Later oracle composition requires its own effect and cleanup acceptance.

## Local verification record

- New comparison capture/kernel cases: 54 passed. Real CPU integration and
  complete public CLI: 24 passed across the main and helper acceptance runs.
- Existing arithmetic, construction, synthesis, source, public API and Algorithms
  regressions: 1255 passed, two optional dependency skips. This includes all six
  file/text executions of the three unchanged original Qmod sources.
- Focused coverage run: 92 passed, eight large already-verified state cases
  excluded from this instrumented subset. The new comparison kernel covers all
  72 statements and 46 branch arcs. This is a function-level coverage claim,
  not a claim that every pre-existing path in the five instrumented modules ran.
- Strict mypy: all 876 package sources and the new user example passed. Current
  and minimum Black 24.1.0, Ruff, API baseline, architecture, dependencies,
  vocabulary, repository language, capability metadata, generated docs, local
  links and ownership checks passed.
- The local default smoke/unit suite stopped at its configured three-failure
  limit: 4550 passed, 121 skipped, the same three previously observed macOS
  native-QFT fusion operation-count failures in
  tests/team/simulation/test_controlled_phase_graph.py (44 versus 28).
  Simulation and those expectations are unchanged; local default is not claimed
  green. Remote CI must establish its own exact-head result.
