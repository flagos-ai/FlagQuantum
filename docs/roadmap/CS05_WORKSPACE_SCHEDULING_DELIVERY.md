# CS-05 workspace lifecycle and scheduling

CS-05 adds two private Compiler schedules behind the existing arithmetic lowering
boundary. It is based on merged CS-04 (`d20de724f1567b0ec23cfd338ffe9959ee98ffb1`).
The user-facing CS-04 workflow and its default conservative result are unchanged.
This is `single_device_fast_path` compiler work; it makes no distributed,
hardware-efficiency or optimal-pebbling claim.

## A fixed program with an actual tradeoff

```python
def pressure(x: int, y: int) -> int:
    t = x * y
    a = t * x
    u = a * y
    return t + u
```

The compiler does not recognize this program by name or fingerprint. The same
analysis and kernels handle the other tested additions, shared values, constants,
unequal widths, zero products and explicit helper calls.

Run the complete developer trial:

```bash
python -m examples.synthesis_workspace
```

It prints both candidates for three unsigned input widths, re-emits each with
exactly its measured workspace capacity, and executes both one-bit coherent
circuits using the existing CPU runtime. An integration test runs this command in
a separate process and checks its table and successful execution. The advanced
diagnostic uses explicit block ports. The normal gate-free named-input CS-04
example remains `python -m examples.algorithms.synthesis_trial`.

All costs below come from the existing Compiler estimator over the complete
arithmetic XOR block, including final cleanup and excluding input preparation.
Inputs and output are disjoint; output width follows full unsigned precision.

| Input bits each | Schedule | Peak workspace | Total qubits | Emitted operations | Emitted depth | Recomputations |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | retain | 6 | 10 | 40 | 35 | 0 |
| 1 | recompute | 5 | 9 | 58 | 54 | 3 |
| 2 | retain | 28 | 39 | 739 | 545 | 0 |
| 2 | recompute | 23 | 34 | 1091 | 860 | 3 |
| 3 | retain | 47 | 65 | 2714 | 2035 | 0 |
| 3 | recompute | 39 | 57 | 4010 | 3115 | 3 |

The existing default compiler preserves the operation counts and depths in all
six rows.

Every emitted and default-compiled candidate uses all declared qubits in these
rows. The same source workload therefore retains a strict space/time tradeoff
after legal simplification; no replacement workload was needed. Tests independently
replay allocation events, succeed at the exact reported capacity, and reject a
capacity one qubit smaller. This establishes actual allocations of this heuristic,
not a proof that each schedule uses the minimum possible workspace.

## Analysis and scheduling rules

`lifetimes.py` verifies the complete definition set, inlines calls with distinct
SSA identities for each invocation, and preserves aliases at returned argument
boundaries. It uses existing HybridProgram/Operation/Value records, not a second
expression IR. Each value has its checked upper bound, direct dependencies and
all use sites. The first multiplication result `t` is used both before and after
the intervening chain. Last use is diagnostic information, not a cleanup proof:
inverse computation can require a dependency after its last forward use.

`lowering.py` owns one allocator and arithmetic emitter. Both schedules reuse
the existing ripple-adder and multiply-accumulate kernels. Addition preserves its
borrowed operand. Multiplier scratch is locally returned to zero. Only generated
X/CX/CCX sequences are reversed; arbitrary unitary phase or cleanup behavior is
never inferred from a block's label.

`retain` preserves the previous default: keep all expression allocations, copy
the result into the XOR target, then reverse the computation. `recompute` erases
non-input operands between forward uses. Before an inverse it restores any
missing dependencies, regenerates the same kernel with their current physical
bindings, and then cleans dependencies introduced only for that inverse. It
keeps final-result dependencies through the XOR copy, avoiding a pointless
restore while the final allocation is still live. Recomputed ancestors may occupy
different qubits; replaying their old controls would be wrong. Final cleanup follows reverse dependency order rather
than rematerialization order, preserving ancestors until descendant cleanup is
finished and avoiding unnecessary final rebuilds.

The zero pool only receives allocations after exact inverse cleanup. Inputs are
immutable basis borrows. Caller-supplied zero workspace remains a precondition;
`fresh_zero` is an allocation obligation. Unknown external states are not proved
zero by this analysis. Nonzero output targets retain `z ^= f(inputs)` semantics.
No measurements, resets, local drop/free, general in-place mutation or arbitrary
state deletion are introduced.

The heuristic is bounded by existing width/node/call constraints, 256 qubits,
100000 emitted operations, 100000 compute/cleanup events and 64 levels of missing
rematerialization dependencies. Budget failure returns no partial circuit and
does not silently fall back to another schedule. A heuristic may cost more
workspace on a different graph; CS-06 must compare actual candidates rather than
assume the name `recompute` guarantees a saving.

## Block integration and evidence

Private `schedule_arithmetic` returns an ArithmeticCandidate containing existing
CircuitIR, lifecycle data, compute/cleanup events, peak reserved workspace and
recomputation count. `lower_arithmetic` returns CircuitIR as before; its private
`scheduling` keyword defaults to `retain`. There are no new stable exports,
Runtime fields, dependencies, numerical kernels or public synth controls.

The admitted block path is a verified construction definition with explicit,
disjoint input/output/zero-workspace bindings, lowered by the shared generator.
The resulting CircuitIR composes through existing Circuit APIs. Arbitrary
external unitary blocks are not admitted as arithmetic merely by declaring
`clean=True`. A general imported-block effect/cleanup contract remains outside
this slice. This preserves one checked block path without introducing a second
trust-flag-based implementation.

Unit scenarios reject changed SSA dependencies, call cycles, unknown unitary
effects, drop, unsupported cleanup assertions, overlapping bindings and unknown
zero state. Function-call tests use the same helper twice, preserve its shared
argument, allocate distinct local results and check the resulting permutation.
The default x*x+y resources remain 18 qubits, 162 operations and depth 123 before
named input preparation, covered by the existing CS-03/04 regressions.

Evidence is deliberately separated by scale:

- One-bit pressure workload: real CPU complex128 execution for every input and
  every two-bit XOR target under both schedules; full complex superpositions
  entangled with an external reference; controlled whole-block execution with
  the control-zero branch as a phase reference, before and after default compile;
  same-block compute/mark/adjoint with a separate analytic phase reference. Full
  vectors are compared without dividing out a global phase. Dirty-workspace
  probability must stay at or below 1e-12.
- Two-bit workload: all sixteen input pairs and all 128 target values per
  schedule are checked by an independent basis interpreter for the emitted
  X/CX/CCX gates, including every workspace bit. Three-bit input pairs are also
  checked with zero and all-one targets. These are gate-permutation checks,
  not claims that 34/39/57/65-qubit dense states were simulated.
- Allocation history and exact-capacity re-emission establish the reported
  physical workspace difference separately from semantic correctness.

The [private boundary record](../api-changes/FQ-SYNTH-CS05-20261029.md) describes
the review scope. Automatic constrained candidate selection, hardware-aware
costs and routed named-result integration remain CS-06 work.


## Local validation

The final implementation passed 40 new scheduling cases and 394 combined
construction, arithmetic, execution, public API and example regressions (the 40
are included in 394), plus 188 Services, experimental namespace and vocabulary
cases: 582 distinct selected cases in total. The example is run as a subprocess.
Branch-aware coverage of `lifetimes.py` and the shared `lowering.py` was 97%
(225/229 statements, 68/72 branches), not whole-package coverage.

Ruff, Black, strict mypy for both production modules and the example,
architecture/dependency checks, the public API baseline, construction acceptance
contract, vocabulary gate, scope, English policy and local documentation links
passed. Stable API contracts and CI configuration were not modified. Fresh PR CI
must validate the submitted head; merged CS-04 results do not validate CS-05.

The final local smoke/unit run stopped at the same three existing macOS QFT
fusion-count failures caused by the unavailable native CPU extension: 4220
passed, 121 skipped, 3 failed after 207.91 seconds with `--maxfail=3`. No Simulation
code or assertions were changed. This is not a green default-suite claim; the
new PR's Linux CI must complete separately.
