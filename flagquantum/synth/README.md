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
0/1 encoding. Chained comparisons, !=/<=/>/>=, Boolean and/or/not, bool parameters,
True/False literals and Python if/loops remain unsupported. Native Qmod syntax
has not expanded.

Run python -m examples.synthesis_predicate for basis and uniform workflows with
no gate design, physical indices or bit decoding. For x*x+y<5 at two bits, the
normal width objective selects 19 total qubits and 14 workspace qubits. The
complete uniform example executes 273 operations at depth 178; these are this
profile's actual costs, not an optimality or hardware claim. The alternative
retained schedule uses 22 qubits and 17 workspace qubits. See the
[comparison delivery](../../docs/roadmap/SYNTH_COMPARISON_DELIVERY.md) for exact
acceptance, raw versus transformed costs and limits.

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
