# CS07-C bounded native source parsing and adaptation

Historical CS07-C evidence below is unchanged. The
[CS07-D checked execution record](CS07_QMOD_EXECUTION_DELIVERY.md) supersedes
its 1/3 execution count and pending Services-admission statements.

Date: 2026-10-30. Baseline: merged #966,
`8722d0fbb37d588d1f0ddf6da586c88acb0cae3d`. Before merge, its exact head
`ffb2c6955b9f4723644c082c34ef6045ada32bb5` completed 57 successful checks and
three configured skips. This stage implements Ecosystem's portion of the
[shared CS07 contract](../api-changes/FQ-SYNTH-CS07-20261029.md), using the
[CS07-B Compiler profile](CS07_REGISTER_PROGRAM_DELIVERY.md).
Classification: single_device_fast_path; private development evidence.

## Existing user workflow and intended outcome

The normal high-level Python workflow remains unchanged. Save this in a Python
source file:

```python
import flagquantum as fq
from flagquantum import synth

def polynomial(x: int, y: int) -> int:
    return x * x + y

program = synth.synthesize(
    polynomial,
    inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
    input_state=synth.InputState.basis(x=3, y=2),
    max_qubits=18,
)
assert dict(program.probabilities(fq.run(program))) == {(11,): 1.0}
```

`python -m examples.algorithms.synthesis_trial` runs the complete existing
basis/uniform/line workflow. CS07-C supplies the private source frontend needed
for future imported examples to share that construction authority. It does not
make ordinary users construct Compiler records or manually decode results.
A public external-source entry and Services checked admission are still CS07-D
work requiring the appropriate API proposal before export.

## Supported source and boundaries

The private `_qmod` package uses standard-library bounded lexing and precedence
parsing, with no new dependency. One qfunc main accepts output qbit/qnum scalar
parameters, optional literal qnum widths, allocate with optional literal size,
X/H/S/Z on qbit, scalar hadamard_transform and fresh assignment. Expressions
admit nonnegative decimal literals, output references, parentheses, +, *, and
literal square. Comments and CRLF preserve source coordinates. No trailing
program, cscope or unknown operation is silently discarded.

The syntax tree is adapter-local. The returned private QmodImport carries only
owned HybridProgram records, expression definitions, inferred-width evidence
and declared source-version context. The public InteropImportResult carries
lowered CoreIR and cannot express these definitions; no second canonical IR,
public schema or adapter registry is introduced. Compiler remains the final
semantic authority. The parser never performs synthesis, execution, source
Python evaluation, file I/O or optional SDK import.

Declared source context is required and pinned to upstream commit
`2bfd835e73b6882fcc515a725098b04766c238e7`. It is not authentication; the retained
fixture manifest independently checks original bytes and license. Current
language documentation was reviewed on 2026-10-30, separately from that older
source snapshot. This is support for the tested subset, not a claim about all
216 files, every standard-library signature or arbitrary future source versions.

### Numeric and lifetime adaptation

Omitted widths resolve at initialization; existing widths stay fixed. Assignment
uses a complete possible-result domain from operand widths, constants and the
admitted monotone operators. For example, source a=4 and b=5 initialize quantum
three-bit outputs; a*b still has domain up to 49 and a six-bit result, despite
the prepared basis value 20 fitting five bits. Positive integer semantics use
unsigned representation with zero fractional digits and at least one bit.
This implements the subset informed by the official
[quantum types/inference rules](https://docs.classiq.io/qmod-reference/language-reference/quantum-types)
and [fresh assignment rules](https://docs.classiq.io/qmod-reference/language-reference/statements/assignment),
not a serialized-layout oracle obtained from a Classiq cloud run.

Pure literal subexpressions fold without executing source. Quantum x*x and
x**2 preserve one borrowed input identity. Fresh assignment initializes a
separate register; constant output storage is not a classical INDEX object.
Output roles, initialize/allocate order, mutation, widths and final live outputs
are checked by the shared Compiler verifier. Existing diagnostics retain their
codes; global model failures receive native entry context when needed.
Unknown Input/Const roles and free/drop/bind behavior are explicitly refused.
The [official variable rules](https://docs.classiq.io/qmod-reference/language-reference/quantum-variables)
provide the role/initialization context; these unsupported effects are not
translated into immutable borrows or zero-workspace reclamation.

A cross-boundary test found an existing CS07-B defect: verification accepted a
wider fresh destination, but lowering passed its full width to an equal-width
arithmetic binding check. The separate Compiler-owned fix binds the same
full-width evaluator to that register's matching low-weight slice. High bits
stay allocated and zero, and both schedule objectives pass full-phase/clean-work
regression. No source declaration is narrowed and binding validation is intact.

### Failures and safety budgets

Reject malformed signatures, unsupported syntax/types/operators/calls, locals,
arrays/views/slices, dynamic flow, helper definitions, signed/fractional values,
in-place operators, unknown symbols, duplicate outputs, repeated initialization,
reads before initialization, narrowing declarations and unknown source versions.
Errors include native filename/line/column; no partially admitted artifact escapes.

Bounds are 262,144 UTF-8 bytes, 4,096 tokens, expression/tree depth 32 and 78
decimal digits per literal; domains stay at most 256 bits. The existing Compiler
also bounds generated nodes and total live width. Oversized comments/source,
long flat expressions, deep nesting, enormous literals, domain overflow and
Hadamard expansion beyond model capacity have tested early refusals. Native
expansion stops before emitting more than 4,096 effect operations; this emission
safety budget is distinct from the Compiler's total model-node limit. All source
must be consumed. No skip, best-effort fallback or input execution is used.

## Original-file compatibility evidence

All source bodies, dimensions and license bytes remain unchanged. Seven source
and license digest checks remain active in the retained corpus tests.

| Original | Parsed to verified construction | Lowered through existing target pipeline | Executed | Complete-state verified | Actual declared/inferred output widths |
| --- | --- | --- | --- | --- | --- |
| X.qmod | Yes | Yes | Yes | Yes | q:1 |
| synthesis_tutorial.qmod | Yes | Yes | No | No | x:3, y:6 |
| multiplication_2vars_example.qmod | Yes | Yes | No | No | a:3, b:3, res:6 |

Counts: source integrity 3/3; parsed 3/3; lowered 3/3; executed 1/3;
full-state verified 1/3. The original X file executes on one actual CPU qubit
with amplitude one at q=1, not a per-file replacement circuit. The other two
originals remain static lowering evidence at 23 and 24 total qubits respectively;
CS07-D must preflight a larger execution budget before full-state acceptance.
Neither declared widths nor original files were reduced for this grid.

Synthetic two-bit tutorial/product, mixed-width constants, phase-bearing
entanglement and input mutation are separately labelled tests. Native and
recorded model producers feed the unchanged numerical conformance consumer from
CS07-B, using the same Compiler verifier, two schedules, target pipeline and
Runtime. Tests compare full complex128 amplitudes and actual mapped clean
workspace within 1e-12. They include real SWAP routing, wider zero-extended
outputs and source numeric significance; all executed cases stay within the
routine 22-qubit grid. They do not inflate original-file counts.

## Validation record

The new acceptance suite passes 99 cases: 82 parser/adapter unit cases,
15 real CPU integration cases and two Compiler padding regressions. The private
parser package has 350 statements and 134 branches, all covered in that focused
suite (100%, not whole-repository coverage). Whole-package strict mypy passes
873 files.

Final expanded regression passed 715 cases with one optional Qiskit skip in
50.79 seconds. It covers CS01-CS07, Services execution/readout, original source
integrity, public admission, interoperability contracts/boundaries, maturity/docs
metadata, current public golden paths and the real subprocess user example.
The skip is not Qiskit execution evidence.

Default smoke/unit reached 4,480 passes and 121 skips before the same three
existing macOS native-CPU QFT fusion-count failures stopped it: 44 observed vs
28 expected (178.99 seconds, 4,392 deselected, --maxfail=3). No Simulation code
or expected counts were changed; full local defaults are not reported as green.
Changed-file Ruff/Black, public API, architecture, dependency, team scope,
vocabulary, repository language, maturity, local links, generated docs and diff
checks passed. Stable contracts and numerical expectations were not weakened.

## Next work and release status

CS07-D must connect native input to existing Services checked-plan/named-reading
and bind the actual result identity and logical/physical ports. Reuse that
execution authority rather than expose a parser-only result as a complete
synthesis Program. Preserve resource constraints, cleanup, read budgets and
normalization refusal. Obtain API review before any public external-source entry.
There is no run_qmod, public qmod namespace, .fq parser, new executor or result
schema in this PR. Stable signatures/exports remain unchanged.

The two independent developer trial records and release approval remain pending.
The [trial guide](SYNTH_DEVELOPER_TRIALS.md) is not human evidence; automated
native correctness cannot fill those records. The public Python synthesis
capability and this private source adapter have separate development-evidence
entries in the capability catalog and separate limitations.
