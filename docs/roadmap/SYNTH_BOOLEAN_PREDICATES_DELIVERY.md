# Composite Boolean predicate delivery

Date: 2026-10-31. Unreleased development evidence. #975 merged as
731172aa795f486fe5a330c93fa84b60276deade after 57 successful checks and three
configured skips on its exact head 2e26f4a9723403e4a9b603db514faaa4d2062a85.
The human instructed continuing the sixth plan, whose second stage adds Boolean
composition after comparisons and phase construction. The
[semantic contract](../api-changes/FQ-SYNTH-BOOLEAN-PREDICATES-20261031.md)
records this narrow slice. Independent trials remain 0/2 and release permission
is false; no GitHub review or trial is fabricated.

## User experience

Run python -m examples.synthesis_boolean_predicate for both value and phase.
Users declare input widths, known preparation, a one-bit computation, its
value/phase effect and resource ceilings. Gates, temporary allocation,
physical indices, uncomputation and bitstring decoding are automatic.

Save this in a source-retaining Python file:

```python
import flagquantum as fq
from flagquantum import synth


def marked(x: int, y: int) -> bool:
    return (x < 2 and y == 1) or not (x == y)


program = synth.synthesize(
    marked,
    inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
    input_state=synth.InputState.uniform(),
    output_mode="phase",
    max_qubits=24,
    memory_limit_bytes=64 * 1024**2,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y"))))
print(program.report["resources"])
```

Phase probabilities remain 1/16. Amplitudes are -1/4 where marked is one and
+1/4 otherwise. Value mode emits (x,y,marked(x,y)) with probability 1/16.
A probability table alone cannot prove the phase; acceptance compares complete
complex amplitudes and interference without removing global phase.

## Semantics and reused implementation

and/or/not accept quantum.uint(1) expressions and numerical INDEX constants
0/1. Optional bool returns, direct one-bit inputs, helpers/constants and aliases
reuse existing rules. Python precedence and operand order are retained on this
domain. All source operands are validated; host short-circuiting does not hide
unsupported calls, wide operands or side effects. No function body is executed.
True/False literals and bool input declarations remain unsupported.

Only Compiler's existing Python frontend changes production behavior: AND
lowers to qint.multiply, NOT to qint.equal against zero, OR to De Morgan
composition. Constant-only Boolean subexpressions fold their exact numeric
values. Quantum values are not folded from prepared input states. Existing
AST/expanded-operation/helper budgets bound all composition work.

The closed SSA schema, verifier, ranges, lifetimes, pure kernels, physical
binding, retain/recompute and finite selection remain authoritative. Value is
XOR into a disjoint output; phase uses the same selected prefix and exact
adjoint to clean workspace. Alias reads do not clone state. No new opcode,
type, contract schema, public signature/export, report field, executor, wrapper
or serialization. Services, Runtime, Core and Simulation are unchanged.

Existing value targets remain eligible under their existing checks; public
phase remains native logical CPU with basis=None/topology=None. Native Qmod
phase, grammar, Boolean control-flow blocks, measurements, bitwise operators,
new comparison kinds, loops, arbitrary angles and general controlled-block
legalization are outside this slice. No old refusal or target guard is weakened.

## Acceptance

46 new Compiler checks cover nested/multi-operand syntax, one-bit inference,
source locations, aliases, helpers, exact constant folding, complete source
validation, wide-operand refusals and construction budgets. The unchanged
Python frontend/comparison suite passed 66 checks.

45 new real CPU/CLI integration checks cover 352 Boolean truth-table basis
executions with both XOR target values under both schedules, 64 public basis
executions for all 16 two-bit pairs in both modes/objectives, and full complex
states entangled with an external reference. They also verify coherent
preparation, helper/constants/qfunc composition, actual costs and plan identity,
workspace zero, phase interference and double application, and early
source/resource/search/memory refusals. The existing line value target is also
checked under both objectives with a nonuniform, phase-bearing RY/RZ input
preparation and an independent closed-form complex reference. Public phase
continues to reject routing/basis requests before capture. The final new focused
suite passed 91 tests.

New Boolean tests execute at most 18 qubits (4 MiB complex128 state buffer),
including the external reference. These are ordinary bounded scalar tests, not
tiny replacements for the separate original-size Qmod acceptance.

Actual complete uniform example costs, including preparation and target passes:

| Mode | Objective / selected schedule | Total qubits | Workspace | Operations | Depth |
| --- | --- | ---: | ---: | ---: | ---: |
| value | width / recompute | 16 | 11 | 394 | 319 |
| value | depth / retain | 17 | 12 | 130 | 97 |
| phase | width / recompute | 13 | 9 | 552 | 457 |
| phase | depth / retain | 16 | 12 | 130 | 97 |

The default example uses width: its planned state buffers are 1 MiB for value
and 128 KiB for phase. Total qubits retain allocated slots even when an
optimization makes a slot unused; actual used counts are 15 and 12 respectively.
Workspace is the source schedule peak, not a claim of minimal physical demand.
Do not equate raw source operations with these actual transformed costs or infer
an optimality/performance claim. Both candidates remain eligible for this logical
example; the objective chooses the measured space/depth trade-off.

The scoped source coverage run passed 112 tests: new executable statements
29/29 and new branch arcs 16/16 covered; aggregate whole-frontend coverage is
93%. The remaining uncovered paths belong to existing frontend cases; this is
not repository-wide coverage. Strict mypy passed 876 package files and the new
public example. Minimum Black 24.1.0, current Black, Ruff, API/architecture/
dependency/team scope, language/vocabulary, capability/generated documentation
and 755 local Markdown-file link checks passed. No signature, candidate contract,
API baseline or historical proposal was regenerated.

Verification environment: CPython 3.12.14, PyTorch 2.14.1, CPU/complex128.
The previous temporary environment had been cleaned; a separate environment
was restored from the existing offline cache without changing project dependencies.
Local default verification passed 4608 tests with 121 skips, stopping at the
same three pre-existing macOS native-QFT count failures (44 versus 28).
Simulation and those expectations are unchanged; local default is not claimed
green. Existing synthesis/construction/arithmetic/phase/native-source regression passed
1472 checks with 21 optional-dependency skips, including all six original-size
public file/text executions of the three unchanged Qmod originals. Their corpus
compatibility count remains 3/3, not a claim about the full library. The separate
public API candidate suite passed 12 checks. These disjoint sets plus the 91
new checks total 1575 passes and 21 skips. Coverage and earlier focused runs
overlap these sets and are not added again. PR/CI status is recorded on the PR
after publication.
The supported profile remains development evidence with independent trials and
release review pending; no GPU/QPU, gradient, scalability or optimality claim.
