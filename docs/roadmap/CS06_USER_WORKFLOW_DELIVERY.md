# CS06-C ordinary synthesis workflow

The existing private Services entry now connects high-level Python arithmetic,
finite constrained selection, a checked actual execution plan and named reading,
including routed nonidentity layouts. It replaces the frozen CS04 conservative
lowering and hash-only reader; it does not create a parallel workflow or public
`flagquantum.synth` namespace. Stable APIs, Runtime implementation/serialization,
dependencies and CI configuration are unchanged.

## Complete ordinary example

```python
import flagquantum as fq
from flagquantum.services import _synthesis as synth

def polynomial(x: int, y: int) -> int:
    return x * x + y

program = synth.synthesize(
    polynomial,
    inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
    input_state=synth.InputState.basis(x=3, y=2),
    max_qubits=18,
)
result = fq.run(program.execution)
print(dict(program.probabilities(result, registers=("x", "y", "result"))))
# {(3, 2, 11): 1.0}
```

Switch to InputState.uniform() to obtain all 16 named input/result tuples with
probability 1/16. Optional qfunc declarations keep the same computation semantics;
ordinary source-retaining Python functions remain the default. There are no
gates, physical indices, workspace allocations, schedule choices or bit decoders
in this workflow. The user declares unsigned input widths, known preparation,
classical constants when present, and optional resource ceilings. The coherent
integer profile preserves borrowed inputs, computes an exact full-width result
into a fresh zero output and cleans its fresh workspace. General arbitrary
Python, free/drop, opaque blocks and general Qmod compatibility are not implied.

Run the complete basis/uniform/line workflow from the repository root:

```bash
python -m examples.algorithms.synthesis_trial
```

The existing subprocess example test checks the printed values. Logical uniform
two-bit polynomial costs include preparation and cleanup: 18 allocated qubits,
10 workspace, 166 operations, depth 124. This is a static actual-executable cost,
not a latency, peak-memory or fidelity measurement.

The same script defines the shared-intermediate pressure function and runs it
with one-bit inputs, uniform preparation, max_qubits=9 and topology="line".
Its four named (x,y,result) tuples have probability 1/4. The actual route contains
SWAPs and ends in a nonidentity layout; the user supplies neither mapping nor
per-width topology table. Read ports come from the verified final layout.

## One source of constraints and actionable recovery

The flat max_qubits/max_operations/max_depth/max_workspace synthesis arguments
are the only user constraint source. Services creates B's existing Constraints
and shared CircuitLimits once. There is no competing constraints parameter or
duplicate resource-limit implementation. ConstructionBudget is independently
bounded and available for expert budget control; a large final ceiling does not
increase a construction safety cap.

For the one-bit uniform pressure function, max_qubits=9 chooses recompute;
max_operations=42 chooses retain. Users change constraints, not schedules.
Selection reports remain available on program.selection for optional inspection;
the normal path uses only program.execution, report and named reading. Large
candidate/evidence fields are omitted from Program's default representation.

SynthesisError exposes B's complete selection result plus immutable flat repair
arguments. A fully checked but infeasible request can be retried explicitly:

```python
try:
    program = synth.synthesize(function, inputs=inputs, max_qubits=8)
except synth.SynthesisError as error:
    if not error.repairs:
        raise
    program = synth.synthesize(function, inputs=inputs, **error.repairs[0])
```

Each repair is feasible for an actually checked candidate and preserves other
ceilings; all repair paths are retried through the real workflow in tests. They
are not global minimum-resource claims. Incomplete construction/search budgets
return no selected success or speculative repair. Unsupported target conversions
and final infeasibility remain distinct B diagnostics. Nothing silently relaxes
the original request.

## Reuse instead of a new execution layer

- Compiler capture, unsigned inference, lifecycle/allocation and retain/recompute
  scheduling remain B/CS05 responsibilities. Services owns declared Python
  parameter labels and preparation composition, and passes names explicitly in
  signature/captured-source order. Reversing the inputs dictionary does not
  reorder the function's quantum arguments.
- B owns basis conversion, optimization, routing, Layout and stage snapshots.
  The optional line profile uses existing CouplingMap.line for the bounded width
  range, with witness-preserving compilation and no final restore. Its default
  supported two-qubit gate basis is x,h,s,sdg,t,tdg,ry,rz,cx,u3,swap when no basis
  was requested. An explicit basis is validated as supplied and never expanded.
  This is a topology profile, not a calibrated hardware/native-gate claim.
- Synthesis invokes existing fq.plan once for the selected submitted IR. It
  performs no circuit execution. Program retains that CircuitIR and the existing
  logical arithmetic block for expert composition. program.execution returns the
  same checked ExecutionPlan after content validation; fq.run(plan) executes
  through the existing no-replanning Runtime path.
- Services binds the B stage snapshot to A's actual sealed program, accounting
  only for the existing runtime_config program-metadata projection. Existing
  CoreIR hashes remain the comparison authority; no serializer/fingerprint
  protocol is invented. Source/stage mutation, changed logical or physical ports,
  basis/constraint changes and mismatched actual programs are refused.
- A remains the sole actual-plan/configuration/profile/layout verifier. The old
  readout identity/profile validator and unconditional routing refusal are
  deleted. Reports use freshly checked actual executable resources; workspace
  limits use B's actual CS05 history. Replacing a Compiler prediction with an
  internally consistent but wrong fake is caught at this real Runtime boundary.

Synthesis now plans once: that deliberately replaces CS04's private no-planning
rule to make final execution agreement concrete. No numerical execution is
introduced during synthesis, execution-property access or reading. Tests count
one planning call and forbid execution during synthesis, then forbid planning,
compilation and execution while reading. Normal users do not build or manage
ExecutionOptions, candidate records, evidence or topology tables.

## Actual probability reading and preserved budgets

The method is `program.probabilities(result, registers=...)`. It returns the
existing named-key probability mapping and its read diagnostics. The earlier
private `read_distribution` spelling is removed; no alias or sampling method is
added. The `fq.run(program)` design review remains separate from this rename.

The sole named reader uses A's physical named ports, validates association with
the same actual plan and reads the supplied result state. Selected-bit, possible
key and amplitude budgets precede normalization scanning and dense marginal
allocation. Nonfinite thresholds and nonnormalized states fail explicitly.
Filtering reports omitted probability and does not renormalize retained keys.
Reordered logical register requests keep their requested tuple order.

Existing canonical plan/source serialization round trips remain admissible; no
object registry is required. Same-sized different programs, changed options,
missing plans, stale source, inconsistent Compiler predictions and substituted
ports are refused. Arbitrary manually forged tensor authentication is not
claimed; reading uses actual supplied state data rather than predicting the
source function's answers.

The initially admitted profile remains one exact CPU torch statevector, static
registered unitaries, no noise/measurements/observations, and A's intact undirected
persistent-layout evidence. Explicit preparation is input-only; declared batch
shapes, unbound symbols, classical conditions and opaque matrices are refused.
Directed/allocated routes or conversions losing SWAP witnesses remain unsupported.

## Technical evidence and release gates

The new integration file adds 26 user scenarios. Together with the migrated 44
CS04 workflow cases, 70 passed. Both direct and line profiles verify named basis/
uniform results, actual SWAP routing, clean physical workspace, constraint-driven
schedule changes, serialization, complete repair/retry and bounded reading.
Bound RY preparation at theta=0.37/1.1/2.0 preserves cos(theta) expectations within
1e-10 and nonzero -sin(theta) gradients within 1e-8 in both profiles. A/B's full
complex-phase and arbitrary XOR/block-composition tests remain required regressions.

Modified synthesis/readout modules have combined statement/branch coverage of
94.93% (254/263 statements, 83/92 branches), not whole-package coverage. Both
modules and the normal example pass strict mypy; architecture checks pass.
Final broader regression passed 757 distinct cases across CS01-CS06, arithmetic,
all existing algorithm examples, Services, plan/result/API contracts, Hybrid IR
and vocabulary. The final targeted 70-case workflow rerun also passed after
clarifying the single-batch preparation diagnostic. Whole-package strict mypy
passed 865 files; Ruff/Black, dependency policy, public API baseline, construction
acceptance, vocabulary, scope, language, links and diff checks passed.

Local smoke/unit (`--maxfail=3`) stopped at the same three recorded macOS native
QFT fusion-count assertions: 4310 passed, 121 skipped, three failed in 202.81
seconds. The two dtype cases of
`test_qft_hadamard_phase_fusion_matches_graph_path_and_reduces_passes` and
`test_product_state_qft_uses_hadamard_phase_fusion` observed 44 passes against
28 in `tests/team/simulation/test_controlled_phase_graph.py`, in this environment
without the native CPU extension. Simulation and the assertions are unchanged.
The local full default tier is not green. Fresh Linux CI on this head is required;
the merged #959 results are not reused.

The old private tests were migrated deliberately: final resource ceilings no
longer double as construction caps, read refusals use A's typed categories and
synthesis may plan once while still never executing. These are not stable public
API changes, and no assertions were weakened to hide a numerical failure.

Completion here is a private technical workflow. At least two independent
developers' real task/time/blocker records and public namespace admission remain
release gates. Agent-run acceptance is not a substitute. CS07 Qmod standard-library
compatibility and .fq artifact semantics remain later work.

See the [private contract](../api-changes/FQ-SYNTH-CS06-C-20261029.md),
[A boundary](CS06_EXECUTION_CONTRACT_DELIVERY.md) and
[B candidate/cost evidence](CS06_CANDIDATE_SELECTION_DELIVERY.md).

Probability-method rename verification: all 70 workflow cases plus the documented
subprocess example passed (71 total), with both examples and tests migrated to
`probabilities`. Whole-package strict mypy passed 865 files; format, architecture,
public API baseline and documentation checks passed. Local smoke/unit again
stopped at the same three native macOS QFT assertions: 4310 passed, 121 skipped,
three failed, 193.51 seconds (`--maxfail=3`). This naming change does not alter
the existing local limitation or replace fresh-head CI.
