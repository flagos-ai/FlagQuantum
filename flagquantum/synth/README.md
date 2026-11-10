# Constrained synthesis

This maintained domain facade re-exports one Services implementation. It adds no
builder, Program wrapper, validation engine or executor. The initial unreleased
candidate supports bounded typed Python programs, source-retaining unsigned
arithmetic and one-bit pi-predicate phases, known basis/uniform preparation,
exact CPU statevectors and the admitted line profile.
Independent developer trial records and final release review are still pending.

The [original-case catalog](../../examples/synthesis/CATALOG.md) separates Qmod
execution coverage from runnable native counterparts and independent Circuit
references. The [native catalog admission](../../docs/api-changes/FQ-SYNTH-NATIVE-CATALOG-20261107.md)
adds direct `X`, `RZ`, `PHASE`, `RZZ`, Bell/GHZ and real amplitude/probability
preparation calls to the existing Python profile. They share Compiler-owned
verification and lowering with Qmod. `synth.X` is a captured gate; the existing
`fq.X` observable constructor is unchanged. See each case for its supported
bindings and the remaining native limitations. The
[native views admission](../../docs/api-changes/FQ-SYNTH-NATIVE-VIEWS-20261107.md)
adds `QArray[QBit]`, checked indexing/slices, bounded builtin `range`/`len`, H/S
and plain writable helper ports. Mutable helpers currently contain only gates,
bounded loops and other mutable helpers, with disjoint actual bit locations.

Root qfunc programs also support `control(bit, lambda: qft(target))`, including
the existing bounded comparison conditions. This reuses the checked controlled
Fourier transform; target and condition inputs must be disjoint. Mutable-helper
control supports phase and the direct controlled-X form below. See the
[controlled Fourier admission](../../docs/api-changes/FQ-SYNTH-NATIVE-CONTROLLED-FOURIER-20261110.md)
and the [Hadamard-test program](../../examples/synthesis/algorithms/quantum_primitives/hadamard_test/main.py).
That program retains its entangled four-bit target as an explicit Output and
selects the ancilla in `program.probabilities`. Selecting readout neither resets
nor frees the target; all five physical qubits remain charged to budgets.

The [all-ones control admission](../../docs/api-changes/FQ-SYNTH-NATIVE-ARRAY-CONTROL-20261110.md)
adds `control(controls, lambda: X(target))` for a numeric register, qubit array,
static element or nonempty slice. Every control bit must be one; a numeric
condition here does not mean nonzero. The target is one distinct writable bit.
The same spelling works inside static mutable helpers. Existing indexed
control-X verification and lowering choose clean scratch and charge it to all
resource budgets. General array-controlled unitary callbacks remain pending. The
[50-control native example](../../examples/synthesis/tutorials/basics/mcx/main.py)
uses a checked exact product-MPS plan, with no dense state allocation. This
[planning admission](../../docs/api-changes/FQ-SYNTH-NATIVE-BASIS-MPS-20261110.md)
is limited to more than32 physical wires and closed I/X/Z/CCX instructions
on the raw IR implicit zero input. The report records the profile and rank bound;
state-buffer, scratch and named-read budgets remain enforced. CX/SWAP and
unproven coherent programs do not qualify for this storage proof.


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

Run `python -m examples.synthesis.tutorials.composition.python_program.main` for nested preparation,
computation, equivalent Qmod and an independent Circuit reference. The UP-01
measurement recorded 18 qubits, 10 workspace, 166 operations and depth 124. The specialized
two-bit Boolean reference uses 9 qubits, one workspace and 15 operations. That
reference requires a hand-derived algorithm; no optimum or equal-cost claim.
See the [UP-01 delivery](../../docs/roadmap/SYNTH_PYTHON_PROGRAM_UP01_DELIVERY.md).

The typed profile supports QBit, unsigned integer QNum and QArray[QBit], Output/Const,
literal basis initialization, allocation, Hadamard preparation, fresh `|=`
expressions using +, *, literal powers 1..8, comparisons, unsigned word bitwise
operations and acyclic positional qfunc calls.
Quantum root ports are Output-only; finite classical parameters follow the binding profile below. The phase, within/apply, adjoint and scalar-control
extensions retain their checked individual scopes. Array views and static loops
follow the native views admission above. Signed/fractional types, Input/free/drop,
nested arrays, host callbacks, dynamic loops/branches and runtime classical
parameters remain outside this Python profile. Existing scalar/Qmod capabilities
retain their own documented scopes.

For example, `res |= a <= 2` coherently computes a one-bit comparison into a
fresh output. `res |= 3 & a`, `res |= a | b` and `res |= a ^ b` compute unsigned
word values with zero extension. `res |= ~a` complements the inferred width of
the quantum encoding: three-bit 6 becomes 1. It does not produce Python's negative
integer `~6`. Outputs must be fresh; `|=` is the capture marker for assignment,
not an in-place OR update. See the
[native expression contract](../../docs/api-changes/FQ-SYNTH-NATIVE-EXPRESSION-VALUES-20261109.md)
and [runnable inversion workflow](../../examples/synthesis/functions/arithmetic/bitwise_invert/bitwise_invert_example/README.md).

Do not combine typed roots with scalar inputs/InputState/constants/functions/
custom preparation or predicate output_mode=phase. A root owns its preparation.
Its default state-buffer ceiling is 512 MiB; explicit limits apply to the actual
checked plan. No new executor, root export or execution unwrap is introduced.

## Static classical parameters

Use the same `synthesize(..., classical_inputs=...)` for native typed qfunc and
Qmod parameters. `float` denotes a scalar and `list[float]` a one-dimensional
real array, bound at synthesis time. Quantum storage still uses Output/QArray.

```python
import flagquantum as fq
from flagquantum.synth import RX, Output, QArray, QBit, allocate, qfunc, synthesize

@qfunc
def encoding(theta: float, q: QArray[QBit]) -> None:
    RX(theta, q[0])

@qfunc
def main(input_0: float, weight_0: float, res: Output[QArray[QBit]]) -> None:
    allocate(1, res)
    encoding(input_0, res)
    encoding(weight_0, res)

program = synthesize(main, classical_inputs={"input_0": 0.37, "weight_0": -0.81})
result = fq.run(program)
print(dict(program.probabilities(result, registers=("res",))))
```

Arrays are snapshotted from built-in lists/tuples; their elements and scalar
bindings must be finite built-in int/float values in ±2**53, excluding bool.
Each array is nonempty and total bindings are limited to 4096 elements. Helpers
can take classical parameters alongside disjoint writable quantum views and use
`len(angles)` in existing bounded static loops. Angle expressions admit +, -, *,
/, unary signs and bounded element access. Real parameters cannot supply indices
or allocation widths. Parameterized mutable helpers retain the gate/loop-only
scope: no allocation, Output escape, adjoint or coherent control. Bindings do not
become runtime tensors or differentiable parameters; synthesize again to rebind.
See the [semantic contract](../../docs/api-changes/FQ-SYNTH-NATIVE-CLASSICAL-20261107.md)
and [native/Circuit QML comparison](../../examples/synthesis/tutorials/basics/qml/comparisons/circuit.py).

Root `prepare_state(probabilities, bound, output)` and
`prepare_amplitudes(amplitudes, bound, output)` also accept named bound real arrays.
Their preparation limit remains 2..256 elements with power-of-two length;
probabilities must sum to one and real amplitudes must have unit norm.
Synthesis validates normalization without repairing it and infers the output
width. Unbound global lists and parameterized Output helpers remain refused.
Run the [complete native vector workflow](../../examples/synthesis/functions/state_preparation/prepare_state_and_amplitudes/main.py)
for host-generated vectors, named sampling and raw signed amplitudes.

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

Run `python -m examples.synthesis.tutorials.getting_started.trial.main` for the complete basis,
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

Run python -m examples.synthesis.tutorials.arithmetic.predicate.main for basis and uniform workflows with
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

Run python -m examples.synthesis.tutorials.arithmetic.boolean_predicate.main for both modes. See the
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

Run python -m tools.synthesis_import_examples.source_trial.main for the complete text
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
`python -m tools.synthesis_import_examples.qmod_comparisons.main`; the complete circuit uses
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

Run python -m examples.synthesis.tutorials.control.phase_oracle.main. Users declare unsigned input
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
Run `python -m tools.synthesis_import_examples.qmod_phase.main`. Arbitrary angles, multibit phase
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
for this fixture. Run `python -m tools.synthesis_import_examples.qmod_helpers.main`.

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
layout. Run `python -m tools.synthesis_import_examples.qmod_const.main`. Source roles can be
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

Run python -m examples.synthesis.tutorials.control.controlled_predicate.main for both examples.
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
automatic. Run python -m examples.synthesis.tutorials.composition.static_loop.main for this sum and a
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

Run `python -m examples.synthesis.tutorials.arithmetic.conditional.main` for both numeric and phase
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
`python -m examples.synthesis.tutorials.arithmetic.interval.main` for both modes and real costs. Uniform
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

Use `python -m tools.synthesis_import_examples.qmod_within.main` or a local .qmod file through the
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
Run python -m examples.synthesis.tutorials.arithmetic.python_powers.main from a saved source file.

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
Run `python -m tools.synthesis_import_examples.native_and_qmod_rz.main` for current Python
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

Run `python -m examples.synthesis.tutorials.arithmetic.boolean_factors.main` for executable native qfunc,
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

`python -m examples.synthesis.algorithms.hamiltonian_simulation.large_pauli.main` demonstrates a small Z evolution
through native qfunc phase intent, Qmod and independent Circuit forms, including
the absolute phase and each form's real workspace cost. It does not claim that
Python qfunc accepts Hamiltonian arrays. The private compact operation is an
implementation detail; normal users continue to call `synthesize` and `fq.run`.

### Exact integer Qmod composition

Qmod can pass an exact integer to a static loop. This public example binds
the count by composing a helper with a literal argument.

```python
import flagquantum as fq
from flagquantum import synth

source = """qfunc twist(steps:int, q:qbit[2]) {
  power(steps) {phase(0.6*q[0]*q[1]);}
}
qfunc main(output q:qbit[2]) {
  allocate(q);hadamard_transform(q);twist(2,q);
}"""
program = synth.synthesize(source, source_format="qmod")
result = fq.run(program)
print(program.report["resources"])
print(dict(program.probabilities(result, registers=("q",))))
```

The |11> amplitude has phase 1.2 radians; each output probability is 1/4.
Run `python -m examples.synthesis.tutorials.composition.integer_composition.main` to compare the Qmod,
static native qfunc and independent Circuit forms. The corpus tools separately
validate externally supplied classical inputs through an internal binding path.
Python qfunc's parameter contract is unchanged. The adapter accepts exact int
and int[] bindings, bounded static loops/if and finite classical-expression time
arrays; it never reads quantum values to select host branches.

Reproduce the unchanged originals with tools/validate_qmod_integer_composition.py:
--case fermi_hubbard_1D --iterations 1 (then 2), --case credit_card_fraud
--data-case 0 (then 1), or --case bb84_implementation, each with --output <path>.
These source and MIT-license-checked runs compare complete complex128 vectors
against independent Circuits, including global phase and numeric bit order.
# Checked Grover composition in the Qmod integration profile

The Qmod adapter supports grover_operator with a checked one-register phase
callback and the built-in hadamard_transform preparation. Top-level int literal
constants and bounded integer specialization share the exact integer profile.
The phase callback cannot capture other quantum values, modify or transfer its
borrowed array, allocate outputs or escape. Validation checks callback structure
even with zero repetitions. General higher-order callback ports, custom
preparation and predicate-output phase_oracle are not implied by this slice.

python -m examples.synthesis.algorithms.grover.grover.main shows runnable Qmod, existing native qfunc
phase composition and independent Circuit forms. The two original Grover
fixtures are verified separately at r=0/1/2, including absolute global phase,
numeric bit order, all workspace bits and resource-budget refusal. No new
native Python Grover callable is exported by this compatibility slice.

## Checked predicate phase oracles

The Qmod integration profile admits a qperm with one Const whole array or flat
fixed unsigned record, one distinct writable single-bit marker and optional
exact scalar integer arguments. Its body must be exactly marker ^= predicate;
the expression must prove a one-bit result and cannot read the marker. All
definitions are checked, including unused definitions. Nominal record identity,
scalar field widths and lexical bindings are retained; field names are not
renamed when a same-named callback formal shadows its own actual register.

phase_oracle accepts a named checked predicate or a two-argument forwarding
lambda. The phase form marks satisfying states by -1. A direct predicate call
XORs any initialized marker, including nonzero and coherent states, using the
existing conditional phase effect between marker Hadamards. There is no new
owned opcode or physical gate template. Capture, input/marker aliases, extra
mutable ports, Output escape and malformed predicates fail closed.

Mathematical negative differences are preserved through comparisons. Unsigned
bitwise arithmetic and positive power-of-two literal modulo are evaluated only
within the existing eight-input-bit predicate bound; monomial, degree, product,
source and final instruction caps are not raised. Larger graph-coloring and
whitebox predicates remain separate resource/semantic blockers, not implied
coverage. grover_search composes uniform preparation with bounded checked Grover
rounds and retains their absolute global phase. General callback ports, custom
preparation and nested/array-valued borrowed records remain outside this slice.

Three unchanged originals are accepted separately: grover_operator.qmod,
oracles_workshop.qmod and grover.qmod (r=0/1/2). The workshop subtraction does not
wrap; the graph original marks >=4, unlike the previous <=4 Max-Cut original.
Run tools/validate_qmod_predicate_oracle.py --case grover --iterations 2
--output /tmp/predicate.json for independent full-state Circuit comparison.
python -m examples.synthesis.algorithms.grover.predicate_oracle.main compares Qmod, existing native
qfunc phase intent and an independent Circuit. It does not add a native Python
phase_oracle/Grover export or a Python QStruct API. The private contract is
docs/api-changes/FQ-SYNTH-UP06-PREDICATE-ORACLE-20261105.md.

Native `qft(target)` applies the positive-sign Fourier transform, including bit
reversal, to a complete initialized writable QNum or QArray[QBit] register. It
shares the checked Qmod lowering and adds no workspace. Whole-register mutable
helpers are supported; partial views and Const targets are refused. See the
[Fourier contract](../../docs/api-changes/FQ-SYNTH-NATIVE-FOURIER-20261109.md)
and [runnable native example](../../examples/synthesis/functions/qft/README.md).

Unsigned `expression % d` supports literal positive power-of-two divisors up to
2**256, using the existing bitwise kernel. General, quantum and nonliteral
divisors remain refused. See the [modulo contract](../../docs/api-changes/FQ-SYNTH-NATIVE-POWER-TWO-MODULO-20261109.md).

For strict static typing, a fixed unsigned port may use standard metadata:
`Output[Annotated[QNum, 5]]` with `from typing import Annotated`. This is the same
five-bit domain as the existing capture spelling `Output[QNum[5]]`. Metadata is
one literal width; signed/fractional annotations remain unsupported.

`unitary(matrix, target)` accepts bounded static complex matrices, using the
existing matrix validator and decomposition into registered gates. Complete
registers, static QArray slices and scalar-bit lists use the same local order:
position zero is the least significant matrix bit. Matrices are literal nested
numeric lists/tuples or immutable global tuples; arbitrary host constructors and
parameterized matrices remain refused. See the
[unitary contract](../../docs/api-changes/FQ-SYNTH-NATIVE-UNITARY-20261109.md).

Typed qfunc fresh assignment accepts coherent conditional values, for example
`res |= a if a < b else b`. The compiler derives a comparison and value-selection
circuit, infers the wider branch width, and cleans its workspace. Nested values
and checked helpers use the existing expression path. Conditions must be one
bit, comparisons, or literal zero/one; host calls and signed/fractional branches
remain refused. See the [conditional-value contract](../../docs/api-changes/FQ-SYNTH-NATIVE-CONDITIONAL-VALUES-20261109.md)
and the [native minimum example](../../examples/synthesis/functions/arithmetic/extremum/minimum_2vars_example/README.md).

`modular_exp(n, a, x, power)` applies coherent modular multiplication to an
initialized numeric target `x` using a distinct exponent `power`. Starting from
`x=1` computes `a**power mod n`; padding states `x>=n` stay unchanged. The base
must be positive and coprime to the modulus. Static integer parameters, target
width 1..5 and exponent width 1..10 use existing validation and lowering. See the
[modular contract](../../docs/api-changes/FQ-SYNTH-NATIVE-MODULAR-EXP-20261109.md)
and [complete native workflow](../../examples/synthesis/functions/arithmetic/modular_exp/modular_exp_example/WORKFLOW.md).

`suzuki_trotter(operator,time,order,repetitions,target)` accepts a static native
`fq.Observable` and finite classical time. Targets include whole registers,
nonempty static slices and lists/tuples of distinct mutable bits in one owner.
Observable index zero addresses the first view element, which is the least
significant bit for a whole numeric register. Spectators retain identity action.
The same target rules apply to fixed-seed `qdrift` calls. Cross-owner, repeated,
invalid or Const locations refuse; zero-time calls still require initialization.
See the [static-view contract](../../docs/api-changes/FQ-SYNTH-NATIVE-PAULI-VIEWS-20261110.md)
and the [native Fermi-Hubbard example](../../examples/synthesis/applications/physical_systems/fermi_hubbard_model_1d/fermi_hubbard_1d/main.py). Orders 1/2 and 1..256 repetitions preserve
declared term order and identity global phase using the shared Pauli lowering.
The result is the specified product formula; no general exact-exponential or
approximation-error guarantee is implied. See the
[native evolution contract](../../docs/api-changes/FQ-SYNTH-NATIVE-EVOLUTION-20261109.md)
and [native example](../../examples/synthesis/functions/hamiltonian_evolution/suzuki_trotter/WORKFLOW.md).

## Checked native unitary regions

The [unitary-region admission](../../docs/api-changes/FQ-SYNTH-NATIVE-UNITARY-REGIONS-20261110.md)
extends the existing markers to pure gates and static writable helpers on
initialized storage. Both Python and the test-only source adapter use the same
Compiler inverse rules. Existing arithmetic-local scopes and Const-only inverse
calls retain their contracts.

```python
import math
import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QNum, allocate, apply, hadamard_transform, phase, qft, qfunc, within

@qfunc
def main(x: Output[QNum[3]], y: Output[QNum[4]]) -> None:
    allocate(x)
    hadamard_transform(x)
    allocate(y)
    with within():
        qft(y)
    with apply():
        phase(y * x**2, 2 * math.pi / 16)

program = synth.synthesize(main, max_workspace=0)
print(dict(program.probabilities(fq.run(program), registers=("x", "y"))))
```

Each pair `(x, x*x %16)` has probability1/8. No gate decomposition, physical
wire numbers, manual inverse or bit-string decoding is needed. The retained
[QCE example](../../examples/synthesis/tutorials/workshops/algo_design_qce_tutorial/algo_design_qce_tutorial_part_i/main.py)
has an independent modular-permutation Circuit reference. Integer literal
powers in phases reuse sparse polynomial lowering without arithmetic scratch.

`with adjoint():` also accepts direct gates and bounded static writable helper
calls. Gate order, constant global phase and matrix conjugate transpose are
preserved. Allocation, Output publication, fresh locals mixed into a unitary
preparation, general host callbacks and ownership changes are refused. Arbitrary
mutable-helper coherent control is a separate boundary. The
[credit-card kernel](../../examples/synthesis/applications/finance/credit_card_fraud/main.py)
checks two nonzero parameter pairs; [randomized benchmarking](../../examples/synthesis/applications/benchmarking/randomized_benchmarking/main.py)
keeps the original sequence and explicitly retains the global phase of SX.
These are local quantum-program checks; full Notebook workflows remain separate.


## Exact signed and binary-scaled numbers

Native scalar annotations support `QNum[width, signed, fractional_bits]`.
Use `Annotated[QNum, (width, signed, fractional_bits)]` for static Python type
checking. A signed register uses two's complement; the encoded integer is
scaled by `2**(-fractional_bits)`. Width-only QNum remains unsigned and unscaled.
Fresh `|=` assignments infer the complete input domain, even when an input was
prepared in a single basis state. Literal `^=` modifies a fitting raw initialized
word; it never numerically adds or changes the scale.

```python
from typing import Annotated

import flagquantum as fq
from flagquantum.synth import Output, QNum, allocate, hadamard_transform, max, qfunc, synthesize


@qfunc
def main(a: Output[Annotated[QNum, 3]], result: Output[QNum]) -> None:
    allocate(a)
    hadamard_transform(a)
    result |= max(3.5, a)


program = synthesize(main)
result = fq.run(program)
print(dict(program.numeric_probabilities(result, registers=("result",))))
```

The exact numeric keys are `Fraction(7, 2)` with probability 1/2, and integers
4, 5, 6 and 7 with probability 1/8 each. The existing `probabilities` method
continues to return raw integer words. `numeric_probabilities` applies the same
bounded named reading and returns immutable exact Fraction keys; verified native
numeric declarations are required. It never infers scale from mutable report
fields or Qmod-prefixed metadata.

Import the typed `synth.min`/`synth.max` markers for quantum extrema; each takes
exactly two positional values and returns a captured QNum. They refuse ordinary
host execution. Python's unshadowed min/max are also recognized during capture,
but their host annotations do not express quantum-valued extrema. After a
wildcard synth import, use `builtins.min`/`builtins.max` in host-only code.

The current slice supports mixed-sign integer bitwise operations/equality,
subtraction/negation, unsigned exact binary multiplication and min/max. Decimal
operands such as 0.1 refuse instead of rounding. Arithmetic reads of scaled
inputs, general signed operators, numeric helpers/locals and signed/scaled
control expressions remain outside this admission. Complete native
quantum examples and independent Circuit references live under
[`examples/synthesis/functions/arithmetic`](../../examples/synthesis/functions/arithmetic/).
Associated Notebook workflows are tracked separately.

## Shared Boolean factors in real phases

Large expressions containing repeated Boolean products can retain those products
as shared predicates instead of expanding every monomial. This is automatic after
an existing expansion-budget refusal; the small-expression path is preserved.
The compiler computes bounded predicates, applies the phase, and uncomputes their
work bits. Existing operator, polynomial and physical resource limits still apply.
This admits no new signed/scaled numeric or controlled-region semantics.

See the [internal semantic contract](../../docs/api-changes/FQ-SYNTH-NATIVE-PHASE-FACTORS-20261110.md)
and [native Facility Location workflow](../../examples/synthesis/applications/logistics/facility_location/WORKFLOW.md).


Native root programs may control a Pauli-axis equatorial rotation with
`control(predicate, lambda: R(theta, phi, target))`. Use `phi=0` for X or
`phi=pi/2` for Y; negative axes are also exact. Other equatorial axes are
rejected rather than rounded. The condition is a scalar bit or a bounded
comparison, including `&`, `|`, `^` and `~` trees of comparisons over at most
eight input bits. The target must be initialized, writable and disjoint from
the condition's registers. Predicate workspace is computed, uncomputed and
included in resource budgets; controlling mutable helper calls remains excluded.
The [native tutorial](../../examples/synthesis/tutorials/basics/quantum_programming/controlled_rotation/README.md)
and its independent Circuit comparison demonstrate the complete workflow.

A Pauli-axis rotation can also use one static array location as its condition:
`control(x[i], lambda: R(theta, pi/2, target))`. A one-bit slice is accepted;
multi-bit conditions and out-of-bounds indices are refused. A gate-only mutable
helper may contain this direct rotation on its checked borrowed scalar ports.
This allows static loops to call a reusable rotation helper, while keeping its
control read-only, its target writable and both underlying registers disjoint.
The [affine rotation example](../../examples/synthesis/functions/linear_pauli_rotations/README.md)
composes `slope*x+offset` on three targets without physical wire indices or work
qubits. Its test-only original and native program both match an independent
block matrix, including coherent input and absolute phase. See the
[indexed control contract](../../docs/api-changes/FQ-SYNTH-NATIVE-INDEXED-PAULI-CONTROL-20261110.md).

## Exact numeric phase reads

Root program `phase(expression, angle)` reads signed and binary-scaled QNum
values with their declared meaning, including negative two's-complement values.
For example, `phase(a*a - a/2 + 0.17, 0.37)` uses the numeric value of `a`, not
its raw unsigned word. The finite coefficient is classical; existing phase
expansion and actual circuit budgets still apply. All encountered reads remain
checked even after cancellation. Numeric comparisons, control conditions,
helper ports and Boolean-factor fallback remain outside this admission.

The [native diagonal QPE example](../../examples/synthesis/functions/qpe/README.md)
combines commuting controlled evolution into a phase expression and then
applies inverse QFT. It preserves `Output[Annotated[QNum, (2, False, 2)]]` for
the fractional phase result. `numeric_probabilities` directly reports exact
phase values 0, 1/2, 1/4 and 3/4, correlated with the four input words at
probability 1/4 each. This concrete composition and its identical flexible-QPE
source counterpart do not add a general callback-based QPE operation.
See the [numeric phase contract](../../docs/api-changes/FQ-SYNTH-NATIVE-NUMERIC-PHASE-20261110.md)
and `tests/integration/test_synthesis_native_numeric_phase.py` for acceptance.

### Reversible arithmetic into initialized outputs

An unsigned root qfunc can XOR a computed value into an initialized scalar:

```python
@qfunc
def main(a: Output[QNum], b: Output[QNum], marker: Output[QBit]):
    allocate(4, a)
    allocate(4, b)
    allocate(marker)
    hadamard_transform(a)
    hadamard_transform(b)
    marker ^= a <= b
```

The target may contain a nonzero or coherent quantum value. The operation means
`target XOR expression`, not replacement. Inputs remain unchanged; actual work
qubits are restored. Expressions reuse the checked unsigned grammar and must fit
the target without truncation. Use `|=` for fresh initialization. Literal `^=`
mask behavior is unchanged. Target/input overlap, uninitialized storage,
signed/scaled expressions, arrays and borrowed mutable-helper expression XOR
are currently refused. Existing root `adjoint` and `within`/`apply` can reverse
expression XOR. The native [QSVT program](../../examples/synthesis/functions/qsvt/main.py)
uses it for reversible comparison; its complete resource report remains visible.

### Reversible preparation on existing storage

`inplace_prepare_amplitudes` and `inplace_prepare_state` apply the existing real
preparation unitary to initialized storage, including nonzero or entangled input.
They allocate no qubits and do not reset the register. The amplitude vector must
be normalized; probability vectors must be nonnegative and sum to one. Adjoint
reverses the complete unitary, rather than just its action on zero input.

```python
import flagquantum as fq
from flagquantum.synth import *

@qfunc
def main(q: Output[QNum]) -> None:
    allocate(2, q)
    hadamard_transform(q)
    with within():
        inplace_prepare_amplitudes([0.5, -0.5, 0.5, 0.5], 0, q)
    with apply():
        phase(0.37 * q, 1)

program = synthesize(main, max_workspace=0)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("q",))))
```

The initial profile accepts a complete writable register and static real vectors
of power-of-two length 2..256. Slices, reordered views, quantum-controlled
preparation, complex amplitudes and Const mutation are refused. Existing
`prepare_amplitudes` and `prepare_state` still initialize fresh output storage.

## Coherent control of a static matrix

Existing `control` and `unitary` compose directly in native Python qfuncs:

```python
import flagquantum as fq
from flagquantum.synth import Output, QBit, allocate, control, hadamard_transform, qfunc, synthesize, unitary

@qfunc
def main(selector: Output[QBit], target: Output[QBit]) -> None:
    allocate(selector)
    allocate(target)
    hadamard_transform(selector)
    control(~selector, lambda: unitary(((-1, 0), (0, -1)), target))

program = synthesize(main)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("selector", "target"))))
print(result.statevector())
```

The zero-selector branch receives minus identity, retaining the relative sign
that probabilities alone cannot reveal. Positive `control(selector, ...)`
selects one. A scalar or indexed selector must be initialized and writable;
selector and target locations must be disjoint. Existing literal/immutable
matrix and target-view rules apply. Selector plus target occupy at most five
matrix qubits, with actual decomposition work still subject to synthesis
budgets. Plain mutable helpers and root inverse/within regions reuse the same
checked unitary effect. Const selectors, arbitrary computed predicates and
multi-bit truth tests remain refused; see the
[semantic admission](../../docs/api-changes/FQ-SYNTH-NATIVE-CONTROLLED-MATRIX-20261110.md).

## Owned private registers

Private storage uses the same quantum types as ports. A constructor initializer
creates a captured handle, not a quantum state. Allocate or prepare it before
use and explicitly drop it at the unconditional root:

```python
import flagquantum as fq
from flagquantum.synth import CX, H, Output, QBit, allocate, drop, qfunc, synthesize

@qfunc
def main(test: Output[QBit]) -> None:
    private: QBit = QBit()
    allocate(test)
    allocate(private)
    H(private)
    CX(private, test)
    drop(private)

program = synthesize(main, max_workspace=0)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("test",))))
```

The test marginal is 50/50. Both physical qubits remain in the Bell state;
`result.statevector()` retains the environment, while `program.registers` exposes
only `test`. Drop does not reset, uncompute, measure or recycle storage. Dirty
private qubits count toward physical width/memory and are distinct from clean
work bits. Missing/duplicate drops, post-drop access, signature/view drops,
Output-helper escape and ownership changes in helpers/loops/branches/unitary
regions refuse. This profile adds no `free` operation. See the
[admission](../../docs/api-changes/FQ-SYNTH-NATIVE-HIDDEN-STORAGE-20261110.md) and
[native overlap example](../../examples/synthesis/algorithms/quantum_primitives/swap_test/main.py).

## Static matrix phase estimation

The native `qpe` subset composes an existing static matrix into controlled powers
and inverse QFT, keeping the numeric counter format explicit:

```python
from typing import Annotated
import flagquantum as fq
from flagquantum.synth import Output, QArray, QBit, QNum, allocate, drop, qfunc, qpe, synthesize, unitary

@qfunc
def main(phase_result: Output[Annotated[QNum, (3, False, 3)]]) -> None:
    state: QArray[QBit] = QArray()
    allocate(1, state)
    allocate(phase_result)
    qpe(lambda: unitary([[-1, 0], [0, -1]], state), phase_result)
    drop(state)

program = synthesize(main, max_workspace=0)
result = fq.run(program)
print(dict(program.numeric_probabilities(result, registers=("phase_result",))))
```

The exact phase value is `Fraction(1, 2)` with probability one; both physical
registers remain charged to budgets.
The current callback is exactly one static unitary call in a zero-argument
lambda, at the unconditional root. The whole numeric counter is one to eight
all-fractional bits, signed or unsigned, initialized by allocate and unused.
General callbacks, flexible powers and nested/inverse QPE remain pending.
See the [matrix-QPE admission](../../docs/api-changes/FQ-SYNTH-NATIVE-MATRIX-QPE-20261110.md)
and the [native matrix example](../../examples/synthesis/algorithms/quantum_phase_estimation/qpe_for_matrix/main.py).
