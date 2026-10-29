# CS06-B finite candidate selection and target costs

Compiler now checks the existing retain/recompute schedules and selects one
fully checked candidate under final constraints. This is a private expert slice:
the CS04 Services facade remains frozen and CS06-C owns ordinary single-entry
integration and routed named reading. No public synth namespace, Runtime
implementation, stable API/schema, dependency or CI configuration changes.

## Reproduce the actual workflow

```bash
python -m examples.synthesis_selection
python -m examples.synthesis_selection_costs
```

The first command prepares coherent inputs, selects recompute under a nine-qubit
ceiling and retain under a 42-operation ceiling, inspects the actual sealed plan
through CS06-A, executes it through existing `fq.run(plan)`, and compares all
complex amplitudes including zero workspace. It prints both complete candidate
rows and both selected schedules. The expert seam has explicit preparation;
this is not presented as the final ordinary-user facade.

The second command emits the complete static matrix and a same-configuration
manual comparison. The [recorded JSON](CS06_CANDIDATE_COSTS.json) contains all 34
rows, source-file SHA256 values, the main baseline revision and construction/
target elapsed observations. Regeneration does not promise identical timing.
It executes no statevector. A separate real test checks the manual and selected
two-bit polynomial on all 16 input components in one coherent state.

## Reuse and ownership

`selection.py` verifies existing HybridProgram definitions, records their existing
semantic identities, infers contiguous input/result ports and invokes the same
CS05 `schedule_arithmetic`. Both candidates retain actual lifecycle/allocation
events, peak workspace and recomputation counts. Unused trailing construction
capacity is trimmed; no allocator, arithmetic kernel or IR is duplicated.

Logical presentation labels come from explicit input_names (anonymous arg0,
arg1 defaults inside Compiler), never from ValueId.scope. The latter is an SSA
namespace; multiple parameters may legitimately share it. C supplies the user's
declared names in argument order. A regression uses a valid same-scope internal
definition to prove the semantic and presentation identities stay separate.

`target.py` composes existing `convert_basis`, Compiler compile/optimization,
persistent-layout routing, optional restore removal, Layout and ASAP scheduling.
Every conversion retains its existing BasisConversionResult; every stage keeps
its CoreIR and recorded content identity. `TargetProduct.validate()` detects
later mutation without recompiling. It does not authenticate arbitrary external
programs or Python transcripts.

CircuitLimits remains the sole authority for width/operations/depth. Constraints
adds the distinct workspace ceiling sourced from CS05 history. ConstructionBudget
holds temporary capacity, operation and candidate-check caps; none are final
ceilings. The target policy and checked stage product are needed because the
existing single-conversion record cannot represent a full target pipeline.
CandidateRow/Selection report partial failures, all checked costs and the choice;
Recovery carries an actual repaired request rather than only a prose hint. These
private records introduce no serialized artifact protocol or registry.

An explicit per-width CouplingMap table represents currently admitted expert
targets. A missing width is refused; there is no silent graph resizing or device
fallback. Fresh graph caches preserve the exact requested width/edges and avoid
cache-history-dependent metadata/identities. This is not a general device
capability selector, and ordinary users will not construct these tables in C.

## Pipeline and execution agreement

The complete scope is preparation, arithmetic and cleanup. Stages are source,
requested initial basis conversion, target compilation/routing, optional layout
restore removal, requested post-conversion, default Compiler normalization, and
ASAP executable order. Final ceilings and leaf-basis/topology checks occur last.
For example, a 140-operation source containing cancelling preparation gates is
admitted under a 40-operation final ceiling after actual legal optimization.
Its independent intermediate construction budget must still accommodate 140.

Compiler predicts instructions; Runtime remains authoritative. The Compiler
prediction retains submitted annotations while recording pass details in its
normalization stage. Runtime's existing canonical sealed program omits the
runtime_config program annotation. These raw hashes are not interchangeable.
Real seam tests explicitly account for that existing metadata boundary and
compare the actual sealed program, then run A's independent plan/configuration/
layout/limit checks. No Runtime canonicalizer or serializer is copied into
Compiler. C must retain this representation binding instead of a raw-hash shortcut.

ASAP can reorder disjoint SWAPs. The pipeline compares the actual routed witness
multiset and complete layout record, replays SWAPs to the final mapping and checks
two-qubit locality. It refuses post-routing conversions that lose witnesses;
this is not an extension of A to arbitrary transformation chains. Real tests
execute nonidentity layouts for both selected schedules and compare full complex
states, nonzero XOR outputs and physically mapped clean workspace.

Static registered input/output preparation is admitted, including bound trainable
rotation tensors. Workspace preparation, batching, observations, measurements,
classical conditions, opaque matrices and unbound symbolic parameters are refused.
For polynomial preparation RY(theta), theta=0.37/1.1/2.0, direct and named-basis
paths match cos(theta) within 1e-10 and its nonzero derivative -sin(theta) within
1e-8. Structural integer constraints remain nondifferentiable.

## Deterministic policy and failures

Ranking version 1 is lexicographic:

- width: allocated qubits, depth, operations, peak workspace, stable candidate ID;
- depth: depth, operations, allocated qubits, peak workspace, stable candidate ID.

Retain has ID 0, recompute 1. Neither timing nor enumeration completion order
affects ranking. Complete ties retain the simpler schedule. All matrix defaults
are checked nondominated on width/workspace/operations/depth. Width remains the
private default: it saves one or five qubits for the pressure cases below, with
the complete gate/depth trade-off visible. This finite review supports this
heuristic only; it is not a universal best default or minimum-width proof.

Saving at most one qubit while doubling operations or depth emits an advisory
with the alternative, including when that alternative is blocked by the current
constraint. A synthetic estimator fixture tests that policy; it supplies no
Runtime numerical or performance evidence. The actual matrix does not cross
that trigger, and depth requests or final limits can choose retain directly.

Every exceeded final ceiling is listed. Construction/candidate/intermediate
budget exhaustion makes evaluation incomplete and returns no selected program,
even if a previously checked row was feasible. No best-so-far success is returned.
Unsupported targets have explicit rows and a distinct top-level failure when no
target can be checked. Unknown gate names, known conversion gaps and normalized
basis drift remain different failures. No gate-set expansion or approximation
repairs the request implicitly. In particular, S/RY conversion gaps require an
explicitly supported requested basis, not an automatic added gate.

Recovery requests are constructed only from actually completed target rows,
relax every violated ceiling together and preserve other constraints. Each is
retried through the real pipeline in tests. They are labeled checked-candidate
values, not global lower bounds. At most two candidates and bounded construction
sizes are examined; non-preemptible Compiler passes do not acquire a wall-clock
timeout guarantee from this stage.

## Full static cost matrix

Cells are `peak workspace / allocated qubits / operations / depth`. Scope is
arithmetic plus complete cleanup, without preparation, identically for both
schedules and the manual baseline. Values are actual final Compiler instruction
costs; larger rows are not dense-state simulation evidence.

| Logical target | Input widths | Retain | Recompute | Width choice |
| --- | --- | --- | --- | --- |
| Addition | 1,2 | 5/11/41/35 | 5/11/41/35 | retain |
| Addition | 2,3 | 6/15/60/51 | 6/15/60/51 | retain |
| Product | 1,1 | 2/5/5/5 | 2/5/5/5 | retain |
| Product | 2,2 | 8/16/88/69 | 8/16/88/69 | retain |
| Square plus add | 1,1 | 4/8/28/25 | 4/8/28/25 | retain |
| Square plus add | 2,2 | 10/18/162/123 | 10/18/162/123 | retain |
| Shared pressure | 1,1 | 6/10/40/35 | 5/9/58/54 | recompute |
| Shared pressure | 2,2 | 28/39/739/545 | 23/34/1091/860 | recompute |

Requested named basis is x,h,s,sdg,t,tdg,ry,rz,cx,u3,swap. This is an explicitly
supported Compiler gate set, not a hardware-native basis claim.

| Named-basis target | Input widths | Retain | Recompute | Width choice |
| --- | --- | --- | --- | --- |
| Addition | 1,2 | 5/11/145/107 | 5/11/145/107 | retain |
| Addition | 2,3 | 6/15/216/159 | 6/15/216/159 | retain |
| Product | 1,1 | 2/5/58/41 | 2/5/58/41 | retain |
| Product | 2,2 | 8/16/677/461 | 8/16/677/461 | retain |
| Square plus add | 1,1 | 4/8/80/61 | 4/8/80/61 | retain |
| Square plus add | 2,2 | 10/18/856/567 | 10/18/856/567 | retain |
| Shared pressure | 1,1 | 6/10/248/155 | 5/9/417/279 | recompute |
| Shared pressure | 2,2 | 28/39/5041/3126 | 23/34/7730/5077 | recompute |

The original manual two-bit polynomial recipe is rebuilt with current Circuit,
adder, compose and adjoint. Under the same target pipeline:

| Target | Manual | Selected synth | Delta synth minus manual |
| --- | --- | --- | --- |
| Logical | 10/18/222/201 | 10/18/162/123 | 0/0/-60/-78 |
| Named basis | 10/18/948/633 | 10/18/856/567 | 0/0/-92/-66 |

These are this recipe's static counts. They do not establish hardware runtime,
memory, fidelity or a fixed improvement percentage for other programs. Individual
construction/target timings are recorded as observations, excluded from ranking.

## Verification and remaining work

The 63 new cases comprise 48 unit/conformance and 15 real integration cases.
The two new modules cover 272/278 statements and 80/86 branches, 96.70% combined;
this is not whole-package coverage. A coverage run initially crashed inside
PyTorch native import before collection; preloading torch before starting the
same pytest/coverage cases completed successfully without modifying or skipping
tests. An earlier normal run also passed the numerical cases.

The real tests exercise constraint-driven choices, all small pressure inputs
and all XOR outputs, routed complex states with nonidentity final layout, clean
workspace, manual/coherent polynomial parity and differentiable preparation.
Conformance replaces the live target transform with recorded results behind the
same selector; it is distinct from actual Runtime numerical evidence.

Final regression passed 731 distinct cases across CS01-CS06 construction,
arithmetic, existing algorithm examples, Services, execution plan/result and API
contracts, Hybrid IR and vocabulary. Whole-package strict mypy passed 865 source
files; both examples passed strict mypy. Ruff, Black, architecture, dependency
policy, public API baseline, construction acceptance, vocabulary, ownership,
language, documentation links, source-hash consistency and diff checks passed.

Final local smoke/unit (`--maxfail=3`) stopped at the same three previously
recorded macOS native QFT fusion-count assertions: 4310 passed, 121 skipped,
three failed in 198.75 seconds. Both dtype cases of
`test_qft_hadamard_phase_fusion_matches_graph_path_and_reduces_passes` and
`test_product_state_qft_uses_hadamard_phase_fusion` observed 44 passes against
28 in `tests/team/simulation/test_controlled_phase_graph.py`, in this environment
without the native CPU extension. Simulation and these assertions are unchanged.
This local full default tier is not green. These results use the final frozen
source after correcting the SSA/presentation naming boundary. Fresh Linux CI on
the new head is required; #957 results are not reused.

C must connect the selector to the existing normal Services synthesize path,
replace the frozen CS04 admission guard, bind Compiler stage identities to A's
actual plan and preserve named-read budgets and missing probability semantics.
It must not expose candidate tables, topology profiles or execution evidence as
required ordinary-user machinery. Independent developer usability trials,
public namespace admission and CS07 Qmod compatibility remain outstanding.

See the [private contract](../api-changes/FQ-SYNTH-CS06-B-20261029.md).
