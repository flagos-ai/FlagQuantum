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
