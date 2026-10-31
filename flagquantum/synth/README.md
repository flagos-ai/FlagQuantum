# Constrained synthesis

This maintained domain facade re-exports one Services implementation. It adds no
builder, Program wrapper, validation engine or executor. The initial unreleased
candidate supports source-retaining Python unsigned arithmetic, known basis or
uniform preparation, exact CPU statevectors and the admitted line profile.
Independent developer trial records and final release review are still pending.

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
qfunc uses the same semantics; it is not required for ordinary functions.

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

Single == and < compare full unsigned values; unequal widths are zero extended.
The logical result is one bit, displayed as integer 0/1 in named tuples. This
computes a flag coherently alongside x/y; it does not measure a condition or
enter Python control flow. Optional bool return annotations require a one-bit
result; parameters still use int/UInt declarations. Explicitly registered helpers
and optional qfunc use the same profile. Existing arithmetic can reuse a flag's
0/1 encoding. One-bit conditions also compose with and/or/not; see below.
Chained comparisons, !=/<=/>/>=, bitwise operators, bool parameters,
True/False literals and Python if/while remain unsupported. Static range loops
use the bounded profile below. Native Qmod syntax
has not expanded.

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
measured costs, exact evidence and refusals. Existing unsupported branching,
measurements, native grammar and independent trial/release gates remain in force.

## Local native source

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
.qmod files, at most 262144 bytes. Native input owns initialization, so inputs,
input_state, constants, preparation and preparation_map must remain None and
functions must be empty. The supported subset is one main, fresh unsigned scalar
outputs, bounded arithmetic and the admitted preparation operations; general
standard-library calls, arrays/views, Input/Const and free/drop remain refused.
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
workflow, or append a local .qmod path. Three unchanged originals have passed
both public forms at 1/23/24 qubits. This is development evidence; independent
trials remain 0/2 and release authorization remains false.

## Predicate phase synthesis

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

Phase currently accepts Python one-bit unsigned returns only, including ==/<
and optional bool return annotations. It requires basis=None and topology=None;
native Qmod phase and explicit basis/routing requests fail before source
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
projective basis conversion, arbitrary angles, controlled-block/within-apply DSL,
Qmod phase, GPU/QPU/gradient admission and full-corpus compatibility remain out
of scope. See [public delivery](../../docs/roadmap/SYNTH_PUBLIC_PHASE_DELIVERY.md)
and the [historical private evidence](../../docs/roadmap/SYNTH_PHASE_ORACLE_DELIVERY.md).
Independent developer trials remain 0/2 and release authorization is false.

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
CPU only. Generic unitary control, within-apply, arbitrary angles and routed
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
