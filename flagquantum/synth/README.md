# Constrained synthesis

This maintained domain facade re-exports one Services implementation. It adds no
builder, Program wrapper, validation engine or executor. The initial unreleased
candidate supports bounded typed Python programs, source-retaining unsigned
arithmetic and one-bit pi-predicate phases, known basis/uniform preparation,
exact CPU statevectors and the admitted line profile.
Independent developer trial records and final release review are still pending.


## Python quantum programs

Prepare state, compute arithmetic and compose helpers inside one qfunc:

```python
import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Const, Output, QNum, allocate, hadamard_transform

@synth.qfunc
def energy(x: Const[QNum], y: Const[QNum], result: Output[QNum]) -> None:
    result |= x*x+y

@synth.qfunc
def main(x: Output[QNum], y: Output[QNum], result: Output[QNum]) -> None:
    allocate(2, x)
    allocate(2, y)
    hadamard_transform(x)
    hadamard_transform(y)
    energy(x, y, result)

program = synth.synthesize(main, max_qubits=24, memory_limit_bytes=2**30)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y", "result"))))
```

All sixteen `(x,y,x*x+y)` outcomes have probability approximately 1/16. The
compiler infers four result bits over the full input domain, implements the
arithmetic and cleans workspace. State preparation now lives in the program;
InputState is still available for the existing pure-function convenience entry.

Use `Output[QNum[2, False, 0]]` and `allocate(x)` when the width is in the
annotation, or `Output[QNum]` and `allocate(2,x)`. Both follow familiar Qmod
concepts. The latter spelling also works with ordinary strict static checkers.
Const inputs can share the same prepared register; fresh Outputs stay disjoint.
All helper calls resolve to captured qfunc definitions, without a public registry
or executing Python bodies. Names/types/operation aliases are checked by identity.

Run `python -m examples.synthesis_python_program` for nested preparation,
computation, equivalent Qmod and an independent Circuit reference. The generic
program uses 18 qubits, 10 workspace, 166 operations and depth 124. The specialized
two-bit Boolean reference uses 9 qubits, one workspace and 15 operations. That
reference requires a hand-derived algorithm; no optimum or equal-cost claim.
See the [UP-01 delivery](../../docs/roadmap/SYNTH_PYTHON_PROGRAM_UP01_DELIVERY.md).

The first typed profile supports QBit and unsigned integer QNum, Output/Const,
literal basis initialization, allocation, Hadamard preparation, fresh `|=`
expressions using +, *, literal powers 1..8, and acyclic positional qfunc calls.
Root ports are Output-only. Signed/fractional types, Input/free/drop, local quantum
variables, arrays/views, host callbacks, loops/branches, explicit phase/control/
inverse and runtime classical parameters remain outside this Python profile.
Existing scalar/Qmod capabilities retain their own documented scopes.

Do not combine typed roots with scalar inputs/InputState/constants/functions/
custom preparation or predicate output_mode=phase. A root owns its preparation.
Its default state-buffer ceiling is 512 MiB; explicit limits apply to the actual
checked plan. No new executor, root export or execution unwrap is introduced.

## Pure-function convenience

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
result = fq.run(program)
assert dict(program.probabilities(result, registers=("x", "y", "result"))) == {
    (3, 2, 11): 1.0
}
```

Run `python -m examples.algorithms.synthesis_trial` for the complete basis,
uniform and routed named workflows. The normal path needs no gate design,
physical numbers, workspace maps, candidate choice or bit decoding. Optional
qfunc supports the unsigned trial subset and the bounded typed program profile
above; explicit real phase and coherent control remain follow-on work.

Services owns capture/preparation composition, the factory-created Program and
named probability reading. Compiler owns existing arithmetic schedules, target
transforms and resource accounting. Runtime owns the existing ExecutionPlan and
execution. `fq.run(program)` validates the product and executes that bound plan;
it never resynthesizes, replans or compiles. All execution overrides must be None.
Changing constraints/targets means synthesizing again. Program.execution is an
expert view, circuit is the explicit expert transformation route, and selection
is optional diagnostics. Manual construction/dataclass fields and serialization
are not public promises. There is no Program.to_ir or general execution protocol.

Do not add numerical kernels, backends, vendor SDKs, dispatch registries or a
parallel compiler here. Implement workflow changes in services/synthesis.py and
export only symbols admitted by the candidate contract. A new accepted target,
root keyword or export requires its own review and executable semantics.

Tests: `tests/api_contract/test_synthesis_run_admission.py`,
`tests/unit/test_synthesis_run_contract.py`, and the existing synthesis integration
suite. See the [implementation admission](../../docs/api-changes/FQ-SYNTH-RUN-IMPLEMENTATION-20261029.md)
and [contract](../../contracts/synthesis-program-run-v1-candidate.json).

The [independent developer trial guide](../../docs/roadmap/SYNTH_DEVELOPER_TRIALS.md)
provides tasks and an unfilled record template for the merged Python entry.
The [CS07 checked source evidence](../../docs/roadmap/CS07_QMOD_EXECUTION_DELIVERY.md)
retains the original private-adaptation record. The
[public source delivery](../../docs/roadmap/CS07_PUBLIC_SOURCE_DELIVERY.md)
implements the separately reviewed file/text entry over the same Program.

## Coherent predicates

```python
def predicate(x: int, y: int) -> bool:
    return x * x + y < 5

program = synth.synthesize(
    predicate,
    inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
    input_state=synth.InputState.uniform(),
    max_qubits=22,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y", "result"))))
```

The six ==, !=, <, <=, > and >= relations compare full unsigned values;
unequal widths are zero extended. Chained comparisons such as low<=x<=high
combine adjacent pairs, capturing each middle expression only once.
The logical result is one bit, displayed as integer 0/1 in named tuples. This
computes a flag coherently alongside x/y; it does not measure a condition or
enter Python control flow. Optional bool return annotations require a one-bit
result; parameters still use int/UInt declarations. Explicitly registered helpers
and optional qfunc use the same profile. Existing arithmetic can reuse a flag's
0/1 encoding. One-bit conditions also compose with and/or/not; see below.
Identity/membership tests, bitwise operators, bool parameters,
True/False literals and Python if/while statements remain unsupported. Static range loops
use the bounded profile below. Native Qmod has a separately bounded binary
comparison profile documented under local native source below.

Run python -m examples.synthesis_predicate for basis and uniform workflows with
no gate design, physical indices or bit decoding. For x*x+y<5 at two bits, the
normal width objective selects 19 total qubits and 14 workspace qubits. The
complete uniform example executes 273 operations at depth 178; these are this
profile's actual costs, not an optimality or hardware claim. The alternative
retained schedule uses 22 qubits and 17 workspace qubits. See the
[comparison delivery](../../docs/roadmap/SYNTH_COMPARISON_DELIVERY.md) for exact
acceptance, raw versus transformed costs and limits.

## Composite conditions

Write a pure one-bit computation using normal Python Boolean syntax:

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

and/or/not accept comparison flags, UInt(1) inputs, one-bit helpers and numeric
0/1 constants. Operand order, nesting and Python precedence are respected on
that domain; multi-bit integer truthiness is rejected. Source is validated in
full, including an operand a host interpreter could short-circuit. A constant
cannot hide an unsupported call or side effect. All accepted expressions are
pure coherent computations; the compiler does not execute Python bodies.

Value mode returns the same expression as a 0/1 result flag; phase mode marks
each amplitude by (-1)**marked(x,y). Phase still requires logical CPU targets.
Every input pair in this example has probability 1/16; complete amplitude and
interference tests verify the phase. Temporary predicate values and workspace
are cleaned automatically through the existing schedules. No Boolean register
type, new executor, manual gates, physical indices or decoding is needed.

Run python -m examples.synthesis_boolean_predicate for both modes. See the
[Boolean delivery](../../docs/roadmap/SYNTH_BOOLEAN_PREDICATES_DELIVERY.md) for
measured costs, exact evidence and refusals. Existing unsupported statement branching,
measurements, native grammar and independent trial/release gates remain in force.

## Local native source

Native compute/phase/cleanup now also accepts bounded `within/apply`; see the
complete example below and the [exact source profile](../ecosystem/_qmod/README.md).

```python
from pathlib import Path
import flagquantum as fq
from flagquantum import synth

program = synth.synthesize(Path("example.qmod"), max_qubits=24)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "result"))))
```

Or use explicit source text, including in a notebook:

```python
program = synth.synthesize(
    """qfunc main(output x:qnum<2>, output result:qnum) {
        allocate(x); hadamard_transform(x); result=x*x+1;
    }""",
    source_format="qmod",
    source_name="polynomial.qmod",
    max_qubits=24,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "result"))))
```

A string is always source text, never a filename. Files must be regular UTF-8
.qmod files, at most 524288 bytes. Native input owns initialization, so inputs,
input_state, constants, preparation and preparation_map must remain None and
functions must be empty. The supported subset is one main, fresh unsigned scalar
outputs, scalar Const/Output helpers, bounded arithmetic and admitted preparation;
general standard-library calls, arrays/views, consuming Input and free/drop remain refused.
See the [adapter profile](../ecosystem/_qmod/README.md) for exact syntax.

Source context defaults to pinned revision
2bfd835e73b6882fcc515a725098b04766c238e7. It declares semantics, not authenticity.
An unsupported explicit source_version fails before file reading. Parsing issues
raise flagquantum.ecosystem.InteropConversionError with structured diagnostics
and source locations; local I/O errors retain their normal exception types.

Native synthesis defaults to a 512 MiB planned-state buffer limit. Override with
memory_limit_bytes explicitly; this is not a process peak-memory guarantee.
Exceeding the limit raises flagquantum.errors.PlanningError before execution.
Python's existing default remains unchanged. Named reading retains its own
max_amplitudes budget, independently of the synthesis memory ceiling.

program.report["register_widths"] lists actual names; the reader default is still
"result". Select registers explicitly if a source uses other names. Both input
forms report whole source_resources and actual resources, source locations,
source_format and effective memory limit/origin. Native reports also retain
source_name/source_version and omit the Python arithmetic_resources field.

Run python -m examples.algorithms.synthesis_source_trial for the complete text
workflow, or append a local .qmod path. Four unchanged originals have passed
both public forms: the baseline three at 1/23/24 qubits and the added integer
comparator at 10 allocated qubits. This is development evidence; independent
trials remain 0/2 and release authorization remains false.

Native source also admits unsigned binary comparisons through Const helpers:

```python
import flagquantum as fq
from flagquantum import synth

source = """qfunc compare(const x:qnum<2>, const y:qnum<2>, output flag:qbit) {
    flag=x<y;
}
qfunc main(output x:qnum<2>, output y:qnum<2>, output flag:qbit) {
    allocate(x); allocate(y);
    hadamard_transform(x); hadamard_transform(y);
    compare(x,y,flag);
}"""
program = synth.synthesize(source, source_format="qmod", max_qubits=24)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y", "flag"))))
```

Each `(x,y,int(x<y))` has probability 1/16. All six unsigned binary relations
are supported; native unparenthesized chains and keyword and/or/not are refused.
One-bit &, |, ^ and ~ compose conditions eagerly; see the native profile.
FlagQuantum also admits phase(x<y,pi) as a native predicate extension; see the
[profile and upstream grammar boundary](../ecosystem/_qmod/README.md).
The named flag remains a live output, with internal workspace cleaned. This
does not change Python predicate phase mode. Run
`python -m examples.synthesis_qmod_comparisons`; the complete circuit uses
10 total/used qubits, five work qubits, 49 operations and depth 40. See the
[delivery record](../../docs/roadmap/SYNTH_QMOD_COMPARISONS_DELIVERY.md).

## Predicate phase synthesis

This section describes the existing bool/int/unannotated predicate profile.
Float-return functions are not admitted; see the UP-00 boundary below.

Use the same high-level predicate with output_mode="phase" to mark input
amplitudes instead of exporting a result bit. Save this in a Python source file:

```python
import flagquantum as fq
from flagquantum import synth


def marked(x: int, y: int) -> bool:
    return x * x + y < 5


program = synth.synthesize(
    marked,
    inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
    input_state=synth.InputState.uniform(),
    output_mode="phase",
    max_qubits=22,
    memory_limit_bytes=64 * 1024**2,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y"))))
print(program.report["resources"])
```

Every input pair has probability 1/16. The amplitude is -1/4 when x*x+y<5 and
+1/4 otherwise; temporary qubits return to zero. **Uniform probabilities alone
do not verify phase.** The acceptance suite compares complete complex amplitudes
and checks interference, including the absolute sign of constant predicates.

Run python -m examples.synthesis_phase_oracle. Users declare unsigned input
widths, known preparation, the one-bit computation, its phase effect and resource
ceilings. They do not design gates, select physical wires, decode bitstrings or
write uncomputation. The default output_mode="value" retains the existing XOR
result and all target options. No decorator is required; qfunc/helpers/constants
work through the same source-capture profile.

The output_mode="phase" option accepts Python one-bit unsigned returns, including ==/<
and optional bool return annotations. It requires basis=None and topology=None;
Qmod with that option and explicit basis/routing requests fail before source
acquisition or capture. A wider integer return is rejected rather than coerced
to truthiness. Existing input_state or mapped static preparation is supported;
preparation stays outside the phase oracle. Memory and construction limits keep
their existing meaning; no numerical execution occurs during synthesis.

Phase Program registers contain inputs only. The existing reader default is
("result",), so select input names explicitly as above; the default raises an
actionable missing-register error. report["output_mode"] is present for both
Python and native inputs. For phase, result_width=0 means no exported result,
scope is actual_cpu_preparation_predicate_phase_cleanup, and arithmetic_resources
is omitted. Use source_resources and actual resources for whole-program costs.

This is unreleased logical-CPU development evidence. General routed phase,
projective basis conversion, arbitrary angles, general controlled-block/within-apply DSL,
arbitrary native Qmod phase, GPU/QPU/gradient admission and full-corpus compatibility remain out
of scope. See [public delivery](../../docs/roadmap/SYNTH_PUBLIC_PHASE_DELIVERY.md)
and the [historical private evidence](../../docs/roadmap/SYNTH_PHASE_ORACLE_DELIVERY.md).
Independent developer trials remain 0/2 and release authorization is false.

### Native sign phase statements

Qmod can instead apply ordered `phase(expression, pi)` statements to initialized
scalar outputs. The expression must reference a quantum register and have a
one-bit unsigned result; this is a bounded sign marker, not arbitrary-angle
phase synthesis. Use the default value mode and read the live named registers:

```python
import flagquantum as fq
from flagquantum import synth

source = """qfunc main(output x:qnum<1>, output y:qnum<1>) {
    allocate(x); allocate(y);
    hadamard_transform(x); hadamard_transform(y);
    phase(x*y, pi);
    hadamard_transform(y);
}"""
program = synth.synthesize(source, source_format="qmod", max_qubits=2)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y"))))
```

The two named results (0,0) and (1,1) each have probability 1/2, within floating
precision. Default costs are 2 qubits, zero workspace, 4 operations (3 H, 1 CZ)
and depth 3. Compiler handles arithmetic selection, exact inverse cleanup and
wire layout; users supply register widths, preparation, phase intent and budgets.
Run `python -m examples.synthesis_qmod_phase`. Arbitrary angles, multibit phase
results, fixed classical phase, helpers and explicit basis/line targets remain
unsupported. The Python-only output_mode option is not needed for source phase
statements. See the [delivery](../../docs/roadmap/SYNTH_QMOD_PHASE_DELIVERY.md).

### Composing native preparations

Native sources can separate preparation and computation into named qfuncs with
fresh scalar outputs. A minimal composition is:

```python
import flagquantum as fq
from flagquantum import synth

source = """qfunc prepare_grid(output x:qnum<2>, output y:qnum<2>) {
    allocate(x); allocate(y);
    hadamard_transform(x); hadamard_transform(y);
}
qfunc prepare_source(output x:qnum<2>, output y:qnum<2>, output energy:qnum) {
    prepare_grid(x,y);
    energy=x*x+y;
}
qfunc main(output x:qnum, output y:qnum, output energy:qnum) {
    prepare_source(x,y,energy);
}"""
program = synth.synthesize(source, source_format="qmod", max_qubits=24)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y", "energy"))))
```

All sixteen `(x,y,x*x+y)` results have probability approximately 1/16. Main's
widths infer to 2, 2 and 4 bits. Compiler still handles gate design, temporary
qubits, cleanup and layout; no gate sequence, physical index or bit decoding is
needed. This source uses 18 qubits, 10 workspace, 166 operations and depth 124,
including preparation and cleanup. Width/depth objectives select the same costs
for this fixture. Run `python -m examples.synthesis_qmod_helpers`.

Helpers accept fixed-width Const and fresh Output scalars. Const actuals must
be initialized and can alias other Const arguments; Outputs remain disjoint and
fresh. Kind/width mismatches, recursion and invalid unused helpers are refused.
At most 32 definitions and Compiler's 4096-node/visit bounds are allowed.
Consuming Input, mutable borrowing, general local scopes, arrays and arbitrary unitary composition
remain separate work. The Python-first workflow is unchanged; no Python
decorator is required by this native compatibility feature. See the
[delivery](../../docs/roadmap/SYNTH_QMOD_HELPERS_DELIVERY.md).

### Computing from live Const inputs

Keep prepared values live while a helper computes a fresh output:

```python
import flagquantum as fq
from flagquantum import synth

source = """qfunc compute_energy(const x:qnum<2>, const y:qnum<2>, output energy:qnum) {
    energy=x*x+y;
}
qfunc main(output x:qnum<2>, output y:qnum<2>, output energy:qnum) {
    allocate(x); allocate(y);
    hadamard_transform(x); hadamard_transform(y);
    compute_energy(x,y,energy);
}"""
program = synth.synthesize(source, source_format="qmod", max_qubits=24)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y", "energy"))))
```

All sixteen `(x,y,x*x+y)` tuples have probability approximately 1/16, with
18 qubits, 10 workspace, 166 operations and depth 124 for this complete example.
Compiler generates arithmetic, selects work storage, cleans it and handles the
layout. Run `python -m examples.synthesis_qmod_const`. Source roles can be
interleaved positionally; the adapter maps them to the same verified Compiler
call ABI used by the previous internal slice.

Const means preserved basis values, permitting phase/entanglement. The helper
may read or phase a Const, but cannot gate it, reinitialize it, compute into it
or pass it as Output; callers may mutate after return. Main remains Output-only.
Bare Const qnum, input/output const combinations, consuming Input and implicit
mutable roles are refused. This adds native source syntax, not Python Const
classes or a required decorator. See the
[delivery](../../docs/roadmap/SYNTH_QMOD_CONST_DELIVERY.md).

## Controlled predicate phase without a control-block API

A one-bit enable value can mask a predicate through existing expression syntax:

```python
def marked(enable: int, x: int, y: int) -> int:
    return enable and (x < y)

program = synth.synthesize(
    marked,
    inputs={"enable": synth.UInt(1), "x": synth.UInt(2), "y": synth.UInt(2)},
    input_state=synth.InputState.uniform(),
    output_mode="phase",
    max_qubits=16,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("enable", "x", "y"))))
```

The control-zero branch keeps its exact amplitude; the control-one branch gains
(-1)**(x<y). The control can be in a superposition or entangled with data. int
reflects Python's operand-valued and on numeric 0/1 inputs; phase synthesis
still verifies a one-bit return. There is no classical branch or input measurement.

The phase lowerer recognizes a terminal product of two one-bit quantum values,
marks their live encodings with CZ (Z for a repeated encoding), and inverts the
same actual compute prefix. It omits the product result and scratch. This
optimization improves already admitted expressions; other phase expressions and
value XOR keep their existing lowering. Earlier predicate work can still dominate
workspace, so a lower peak is not guaranteed for every function.

For `return enable and flag` with two UInt(1) inputs, uniform phase synthesis
fits max_qubits=2, max_workspace=0 and max_operations=3, including two Hadamards
and one CZ. The raw general prefix baseline uses four allocated qubits and
seven gates; the direct raw marker uses two and one. Source and actual complete
costs are distinct. No hardware timing or global optimality claim follows.

Run python -m examples.synthesis_controlled_predicate for both examples.
Uniform named probabilities alone cannot certify controlled phase; complete
complex-state and interference acceptance does. Public phase remains logical
CPU only. Generic unitary control, general within-apply, arbitrary angles and routed
phase remain separate work. See the
[controlled-predicate delivery](../../docs/roadmap/SYNTH_CONTROLLED_PREDICATE_DELIVERY.md).

## Static loops without manual expansion

Use an explicit classical configuration for iteration counts:

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
```

This computes x+3*y, with (x,y,result) equal to (0,0,0), (0,1,3), (1,0,1)
and (1,1,4), each with probability 1/4. Result width and temporary qubits are
inferred. Quantum loop-carried values become fresh SSA values; input registers
and prior aliases retain their original encoded values. Workspace cleanup,
choice between retain/recompute, target checking and named decoding remain
automatic. Run python -m examples.synthesis_static_loop for this sum and a
loop-built one-bit predicate in phase mode.

The supported loop is for name in range(...), with 1-3 nonnegative classical
integer arguments and positive step. Bounds use constants and pure classical
computations, never quantum values or known basis samples. Inputs and index
are read-only. Initialize each carried local before the loop and assign it once
per iteration with name = expression. New body locals and indices are scoped to
the iteration. Outside loops the existing single-binding rule remains in force.

There is one active loop, including helper expansion, and at most 256 expanded
iterations per capture across helpers/sequential loops. Existing source-node,
operation, width and construction budgets still apply. Nested/while loops,
augmented assignments, break/continue, loop else, branches and body return remain
unsupported. Empty loops preserve old bindings and validate their bodies before
discarding provisional SSA; unsupported hidden calls cannot escape. Unused
classical definitions are omitted at capture completion without deleting quantum
operations, preserving zero-workspace phase masks despite loop scaffolding.

This is compile-time unrolling through the existing graph and kernels. There is
no runtime loop, new API, executor, type or native Qmod grammar. Costs depend on
the generated graph; algebraic simplification and equality of looped/manual gate
sequences are not promised. See the
[static-loop delivery](../../docs/roadmap/SYNTH_STATIC_LOOPS_DELIVERY.md) for
scope, measured resources and complete-state acceptance.

## Select values coherently

Pure conditional expressions select an unsigned value for every input without
measuring the predicate. For example, save this in a source-retaining file:

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

Every `(x, y, min(x,y))` tuple has probability 1/16. Result width is two bits,
with 11 total qubits, five workspace qubits, 98 operations and depth 80 including
preparation. In this example both width/depth objectives select retain; these
are measured candidate costs, not an optimality or performance guarantee.
No decorator, gate sequence, physical map or decoding is required.

A predicate must be a one-bit UInt value, an existing comparison/Boolean
expression, or classical integer 0/1. Branches can be unsigned expressions,
unequal widths or literals; values zero-extend without truncation. Both branches
are validated, even if a classical guard picks only one. Already captured
quantum branch work is conservatively retained and cleaned. Arbitrary Python
side effects and short-circuit execution are unavailable.

Nested selections, helpers, optional qfunc and static loop assignments compose.
One-bit branches produce a one-bit result, so
`return (x<y) if enable else (x==y)` can use `output_mode="phase"` with the
existing exact CPU Program/run path. Complete amplitudes establish this phase;
uniform probabilities alone cannot. Statement-level if/else, mutable quantum
branches, generic controlled blocks and measurement feedback remain unsupported.

Run `python -m examples.synthesis_conditional` for both numeric and phase
examples. See the [delivery record](../../docs/roadmap/SYNTH_CONDITIONAL_EXPRESSION_DELIVERY.md)
for independent references and measured costs. Trial/release, target and native
Qmod restrictions retain their existing gates.

## Intervals and exclusions

Write ordinary unsigned comparisons, including chained bounds, in a retained
Python source file:

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

The marked values are 1, 2, 4, 5 and 6; each input/result pair has probability
1/8. Users declare input widths, classical constants, preparation and budgets.
The compiler handles comparator choice, one-bit Boolean composition, gates,
physical maps and cleanup, and the Program reads named values. No decorator is
required. Middle expressions in chains are captured once; all parts validate,
even after a classically false pair. Source effects are not short-circuited or
executed on the host. Bool literals/parameters, identity/membership and signed
or fixed-point comparisons remain unsupported.

Use `output_mode="phase"` in the same call to mark each accepted basis amplitude
by -1, with only input registers returned. Run
`python -m examples.synthesis_interval` for both modes and real costs. Uniform
probabilities alone do not certify phase; acceptance compares full amplitudes
including external reference entanglement and clean workspace. See the
[delivery record](../../docs/roadmap/SYNTH_COMPARISON_CHAINS_DELIVERY.md).

## Native temporary conditions without exported flags

```python
import flagquantum as fq
from flagquantum import synth

source = """qfunc mark(const x:qnum<2>, const y:qnum<2>) {
    flag:qbit;
    within { flag=x<y; } apply { phase(flag,pi); }
    within { flag=x==y; } apply { phase(flag,pi); }
}
qfunc main(output x:qnum<2>, output y:qnum<2>) {
    allocate(x); allocate(y);
    hadamard_transform(x); hadamard_transform(y);
    mark(x,y);
}"""
program = synth.synthesize(source, source_format="qmod", max_qubits=10)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y"))))
```

Every pair has probability 1/16; full amplitudes are (-1)**int(x<=y)/4.
Only x/y are named results. Compiler generates gates, checks read-only apply
dependencies, reverses each selected computation, proves zero and reuses the
temporary slot. The two scopes fit ten qubits, including six internal work bits;
simultaneously preparing both flags needs eleven. Width and depth objectives
both preserve these measured costs; scope orders are not automatically rewritten.

Use `python -m examples.synthesis_qmod_within` or a local .qmod file through the
same entry. Local declarations are function-level; within prepares fresh
quantum-dependent unsigned expressions, and apply permits phase or Const-only
helpers. Sequential blocks may reuse the source name. Bare qnum infers once;
explicit widths preserve zero extension. Gates/writes in apply, literal/local
allocation, lexical nesting, manual free/drop and consuming Input are refused.
This bounded source profile adds no Python decorator or public API. Existing
Python predicates remain the concise entry when importing source is unnecessary.
See the [delivery and limits](../../docs/roadmap/SYNTH_QMOD_WITHIN_DELIVERY.md).

## Positive literal powers in Python

```python
import flagquantum as fq
from flagquantum import synth


def cube(x: int) -> int:
    return x**3


program = synth.synthesize(
    cube,
    inputs={"x": synth.UInt(2)},
    input_state=synth.InputState.uniform(),
    max_qubits=24,
    memory_limit_bytes=256 * 1024**2,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "result"))))
```

The pairs (0,0), (1,1), (2,8) and (3,27) each have probability 1/4.
Positive literal exponents 1..8 share existing multiplication synthesis and
cleanup with native powers. No decorator, temporary register or multiplication
chain is required. Users still declare input domains, preparation and budgets.
Configured/computed exponents, bool, zero, negative, fractional and larger values
are refused. See the [delivery](../../docs/roadmap/SYNTH_PYTHON_POWERS_DELIVERY.md).
Run python -m examples.synthesis_python_powers from a saved source file.

## Unified program contract and compatibility

The [UP-00 contract](../../docs/api-changes/FQ-SYNTH-UNIFIED-PROGRAM-UP00-20261102.md)
keeps familiar Qmod concepts as the program design baseline: qfunc, quantum
roles, allocate, expressions, phase, calls and control. It fixes their meaning
and staged admission. UP-01 now exports Output/Const/QNum/QBit and the bounded
preparation/arithmetic/call subset above; arbitrary effectful Python bodies
remain refused. Existing `.qmod` sources do not need conversion
into a new Python syntax. Unsupported constructs fail rather than being guessed.

The unmerged float-return radians prototype from #1003 is withdrawn. Existing
int/bool/unannotated Python predicates still mean pi phase; general explicit
native phase programs remain a gap for UP-02. The planned phase operation is
`phase(expression, coefficient=1.0)`, not a new return-type convention.

Current RZ compatibility preserves the Qmod statement:

```python
import flagquantum as fq
from flagquantum import synth

program = synth.synthesize(
    "qfunc main(output q:qbit) {allocate(q);H(q);RZ(1.9,q);}",
    source_format="qmod",
    memory_limit_bytes=1024**2,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("q",))))
print(result.statevector())
```

This is a coherent variant with H, not the unmodified pinned RZ original.
Amplitudes are exp(-0.95i)/sqrt(2) and exp(0.95i)/sqrt(2); probabilities are
approximately 1/2 each. Decimal PHASE/RZ limits and gate-target restrictions are
in the [compatibility delivery](../../docs/roadmap/SYNTH_LITERAL_RZ_DELIVERY.md).
Run `python -m examples.synthesis_native_and_qmod_rz` for current Python
high-level arithmetic, matching Qmod and the independent RZ Circuit reference.
The example states the native phase gap rather than calling Circuit high-level.

### Shared Boolean computations and original QAOA costs

Native qfunc arithmetic and Qmod reuse the same Compiler Boolean candidate.
For example, the user declares a product and a phase without choosing an AND
ladder, scratch wires or cleanup gates:

```python
from flagquantum.synth import (
    Output, QBit, QNum, qfunc, allocate, hadamard_transform,
    within, apply, phase, synthesize,
)

@qfunc
def main(a: Output[QBit], b: Output[QBit], c: Output[QBit], d: Output[QBit]):
    allocate(a)
    allocate(b)
    allocate(c)
    allocate(d)
    hadamard_transform(a)
    hadamard_transform(b)
    hadamard_transform(c)
    hadamard_transform(d)
    with within():
        selected: QNum[1] = a * b * c * d
    with apply():
        phase(selected, 0.37)

program = synthesize(main, max_qubits=7, max_workspace=3)
```

Run `python -m examples.synthesis_boolean_factors` for executable native qfunc,
Qmod and independent Circuit forms. Both synthesized programs use seven qubits,
17 operations and depth 14, including three clean workspace qubits. This does
not add masked-integer expressions to the Python qfunc frontend.

For original Qmod real costs, the flat polynomial path is retained. When its
bounded expansion fails, the adapter can preserve shared products of signed bit
literals using existing arithmetic calls. The Compiler verifies their inverse
cleanup and chooses their physical realization. The original TSP and facility
location files keep their bytes and layer counts; reproduce their independent
full-state checks with `python tools/validate_qmod_phase_factors.py --case
facility_location --scale .003 --output /tmp/facility.json` (one shell line).
See [the bounded profile](../../docs/reference/KNOWN_LIMITATIONS.md) for the
factor, active-bit, expression-depth and resource limits. This is a local
integration capability pending the complete 216-original gate.

### Large Pauli evolution originals

The original 551-term, 12-bit Pauli workloads now use compact owned effects
and bounded Compiler lowering. qDRIFT retains all 2000 seed-zero samples.
Use the same public workflow with either unchanged local file:

```python
from pathlib import Path
import flagquantum as fq
from flagquantum import synth

program = synth.synthesize(
    Path("hamiltonian_evolution.qmod"),
    max_qubits=12,
    max_workspace=0,
    memory_limit_bytes=64 * 1024**2,
)
result = fq.run(program)
print(program.report["resources"])
print(result.statevector())
```

Run `python tools/validate_qmod_large_pauli.py --case hamiltonian_evolution
--output /tmp/qdrift.json` on one shell line for the pinned fixture, independent
sample sequence and full-state Circuit reference. The companion case is
`example_exponentiation`. Source files and their MIT license are hash checked.

`python -m examples.synthesis_large_pauli` demonstrates a small Z evolution
through native qfunc phase intent, Qmod and independent Circuit forms, including
the absolute phase and each form's real workspace cost. It does not claim that
Python qfunc accepts Hamiltonian arrays. The private compact operation is an
implementation detail; normal users continue to call `synthesize` and `fq.run`.

### Exact integer Qmod composition

The Qmod entry can bind classical loop counts without floating-point coercion:

```python
import flagquantum as fq
from flagquantum import synth

source = """qfunc main(steps:int,output q:qbit[2]) {
  allocate(q);hadamard_transform(q);
  power(steps) {phase(0.6*q[0]*q[1]);}
}"""
program = synth.synthesize(source, source_format="qmod", classical_inputs={"steps": 2})
result = fq.run(program)
print(program.report["resources"])
print(dict(program.probabilities(result, registers=("q",))))
```

The |11> amplitude has phase 1.2 radians; each output probability is 1/4.
Run python -m examples.synthesis_integer_composition for the executable Qmod,
static native qfunc and independent Circuit forms with real workspace costs.
Python qfunc's parameter contract is unchanged. The adapter accepts exact int
and int[] bindings, bounded static loops/if and finite classical-expression time
arrays; it never reads quantum values to select host branches.

Reproduce the unchanged originals with tools/validate_qmod_integer_composition.py:
--case fermi_hubbard_1D --iterations 1 (then 2), --case credit_card_fraud
--data-case 0 (then 1), or --case bb84_implementation, each with --output <path>.
These source and MIT-license-checked runs compare complete complex128 vectors
against independent Circuits, including global phase and numeric bit order.
