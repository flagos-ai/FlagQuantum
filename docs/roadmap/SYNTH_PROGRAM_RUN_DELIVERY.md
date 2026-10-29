# Synthesis Program execution admission

The reviewed #963 scope is implemented on this branch. Normal users import the
maintained synth facade and pass the complete synthesis product directly to
fq.run. Independent developer trials and final release review remain pending;
this is an unreleased implementation candidate, not a general Qmod/backend claim.

## Complete user workflow

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
print(dict(program.probabilities(result, registers=("x", "y", "result"))))
# {(3, 2, 11): 1.0}
```

Run `python -m examples.algorithms.synthesis_trial` for the documented basis,
uniform and real line-profile examples. Both ordinary functions and optional
qfunc use the same verified arithmetic profile. Users declare unsigned domains,
known preparation, classical constants and optional constraints. They design no
gates, workspace map, schedule, physical layout or bit decoder.

Synthesize returns the complete Program containing its circuit, brief report,
checked execution and logical output bindings. It does not return only a plan.
The normal path needs neither ExecutionOptions nor execution-property access.
execution remains an expert view and circuit the explicit expert transformation
route; a changed circuit needs resynthesis/revalidation rather than stale reading.

## One type and one execution authority

The existing Services implementation moved from `_synthesis.py` to `synthesis.py`.
The seven exports in flagquantum.synth are those same Services objects. Program
is defined in flagquantum.services.synthesis, not wrapped or subclassed by the
facade. All live imports moved with it, and the unreleased private module was
removed without an alias. Private readout/execution helpers remain at their
owning boundaries; Compiler, arithmetic kernels and Runtime implementations are
unchanged.

The root facade recognizes the loaded canonical owner without importing Services
on ordinary Circuit calls. It refuses user subclasses, then rejects every
non-None execution override before accessing the checked plan. It obtains
program.execution, which performs existing product/stage/port validation, and
continues through the existing ExecutionPlan branch. No recursive public run,
replanning, compilation, synthesis or custom executor is introduced.

The unchanged result type is ExecutionResult and result.plan remains the actual
ExecutionPlan. Existing plan serialization/replay is reused; Program/.fq
serialization is not added. No new registry, executable protocol, Core artifact
type or to_ir coercion is needed. Existing non-Program circuit coercion and
Circuit/IR/plan/Ising/Lindblad behavior remain on their existing paths.

## Bound execution and compatibility

For a Program, options, outputs, noise_model, compiler, target, target_qubits,
shots and name must all be None. Passing an equivalent options object is also
refused. These controls already belong to the checked program's execution; users
must synthesize a new request for different constraints or targets. Existing
controls for other input types retain their behavior.

The initial profile remains exact single-batch CPU torch statevectors, registered
static unitaries, bounded unsigned arithmetic/preparation and admitted undirected
persistent-layout evidence. Directed/allocated/lost-witness routes, noise and
measurements remain outside the profile. Named probabilities preserve A's actual
physical ports, normalization refusal, pre-scan budgets and omitted mass without
renormalizing. C's bound scalar preparation gradients remain intact.

The aggregate run annotation adds only SynthesisProgram. All keywords/defaults
and the return annotation are unchanged. The current execution-plan and output
candidate contracts carry that precise addition. The historical baseline,
proposal JSON and previous presence/Ising approval records are retained.

The dedicated admission checker verifies the narrow delta against the historical
before/after signature, canonical owner, exact facade exports, source functions,
rejection policy and actual method signatures. Negative tests reject changed
authorization/release/trial claims, arbitrary input types, extra exports, wrong
owners/namespaces, non-boolean flags, missing override controls and unknown fields.
This is an explicit extension of protection, not a snapshot-regeneration bypass.
The historical root-disposition file remains a record of original root symbols;
the seven new domain exports are registered in their own maintained-extension
candidate rather than misclassified as historical root migrations.

## Evidence and rollout

The first 135 focused cases passed: production admission, exact contract mutation
tests, historical proposal preservation, existing user workflows, plan/public API
compatibility and the real subprocess example. Production admission tests compare
complete complex phases, plan identities, resources, names and zero workspace
for both schedules and logical/nonidentity line layouts. They forbid overrides
before resolution and reject stale products before numerical dispatch.

Recorded existing plan replay replaces the checked plan behind the same actual
fq.run consumer. Planning, compilation and synthesis are disabled during the
replay; there is no test-only replacement for the production entry. Numerical
references remain independent full-state comparisons. A fresh-process test
verifies ordinary Circuit import and execution load neither synth nor Services.
All existing C gradient cases now exercise fq.run(program) directly.

Whole-package strict mypy passed 868 files; the normal example also passed strict
mypy. Public API, architecture and initial formatting checks passed. A separate
optional strict invocation over the entire legacy public_api_snapshot tool
reports five pre-existing errors at lines 119/292/298/299/366, reproduced from
the main version. They are outside the new validator and no ignore/check bypass
was added. The validator's functional and mutation tests pass; the production
package type check is green.

Expanded regression passed 1,019 cases with eight skips in 152.11 seconds. It
covers arithmetic/construction, Services, API contracts, examples, Module runtime,
Ising and Rydberg presence compatibility alongside the new admission cases.
Skips are not evidence for unavailable hardware capabilities.

The default smoke/unit run reached 4,346 passes and 121 skips before stopping at
three existing native-CPU QFT fusion-count failures (221.81 seconds; 4,354 tests
deselected). Those tests observe 44 operations instead of 28 on this macOS
installation without the native CPU extension. No Simulation code or numerical
expectations were changed; the full local default tier is not reported as green.

Architecture, team ownership, public API, dependency, construction capability,
vocabulary, repository language, documentation links and changed-file formatting
checks passed. Capability-maturity and documentation-source validation also
passed, with 35 associated unit tests. The capability catalog records this profile as development
evidence, with independent trials and release approval explicitly pending.

Implementation authorization is recorded from the actual user instruction to
merge #963 and continue. It does not claim a fabricated GitHub review, completed
independent trials or release approval. The machine candidate keeps
release_authorized=false and human_trial_records empty. Two independent users
must run the implementation branch's basis/uniform/constraint/line/recovery tasks,
record commits, elapsed work, code, help and blockers, and provide actual outputs
before final activation/release review. Automated correctness is not that evidence.

See [namespace ownership and examples](../../flagquantum/synth/README.md),
[implementation authorization](../api-changes/FQ-SYNTH-RUN-IMPLEMENTATION-20261029.md),
[candidate contract](../../contracts/synthesis-program-run-v1-candidate.json) and
[historical proposal](../api-changes/FQ-SYNTH-RUN-ADMISSION-20261029.md).
