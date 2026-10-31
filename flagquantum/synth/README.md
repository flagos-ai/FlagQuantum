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
records private adaptation and original-size CPU verification for three retained
Qmod files. A [public source proposal](../../docs/api-changes/FQ-SYNTH-QMOD-ADMISSION-20261031.md)
specifies the additional input journey for review; no public file/text source
entry is implemented by that proposal.
