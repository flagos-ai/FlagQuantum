# CS06-A execution consistency contract and real seams

This is the first CS-06 slice from revision six, based on merged CS-05 at
`4763937d3cc060212cf45d2c9ac6d7cf482ca940`. It checks the executable plan that
Runtime will run. Candidate selection remains CS06-B and normal user-facade
integration remains CS06-C. No public synth namespace or stable API is changed.

## Why the submitted circuit is insufficient

Runtime planning can optimize a submitted circuit again. The preparation probe
already found that a circuit converted to h/t/tdg/cx/x and routed with optimization
disabled was later executed with additional `u3` gates and different counts.
Numerically correct output alone did not establish the requested basis or costs.

The new real regression makes the problem small and reproducible: h followed by
s is initially in the h/s basis, but `fq.plan` produces one `u3`. A source-only
check would accept that target; the plan check refuses it before execution. An
explicitly different basis request is a separate check, never an automatic
widening of the original target.

A second regression starts with two logical instructions, h and a nonlocal cx.
Its actual routed executable needs three instructions. A two-operation ceiling
is rejected and a three-operation ceiling is retried and passes. Limits are
applied to the exact whole executable, including input preparation, arithmetic,
routing and cleanup, rather than the arithmetic block alone.

## Runnable expert workflow

```bash
python -m examples.synthesis_execution_contract
```

The [complete example](../../examples/synthesis_execution_contract.py) uses the
existing `fq.plan` and `fq.run(plan)` entry points:

```python
from flagquantum.compiler._construction.constraints import CircuitLimits
from flagquantum.services._synthesis_execution import inspect_execution

# plan is the existing fq.plan(...) product, not a new execution manager.
checked = inspect_execution(
    plan,
    limits=CircuitLimits(max_qubits=3, max_operations=3, max_depth=3),
    basis=("u3", "swap", "cx"),
    coupling=topology,
    registers={"input": (0,), "result": (2,)},
)
result = fq.run(checked.plan)
resources = checked.verify_result(result)
```

Here `topology` is the existing three-qubit line CouplingMap. The example routes
a phased Bell preparation with a real SWAP, removes the final layout restore
through the existing Compiler API, and obtains final layout (1,0,2). Inspection
projects logical input port 0 onto physical port 1 and logical result port 2 onto
physical port 2. The state must equal `(|000> + i|011>)/sqrt(2)` in that
physical order, comparing every complex amplitude without discarding global
phase.

Expected output includes basis-drift rejection, physical registers
`{'input': (1,), 'result': (2,)}`, three qubits/three operations/depth three, and
successful nonidentity routing/phase verification. A subprocess integration test
runs the documented command. This is an expert seam demonstration; normal users
still use the unchanged CS-04 named preparation and reading path.

## One authority per responsibility

Compiler owns the new pure `CircuitLimits` record. It checks allocated qubits,
final operations and depth using ResourceEstimate and returns existing Diagnostic
records for all exceeded ceilings. Zero operations/depth is valid for an empty
program; qubit capacity remains positive. Limits reject booleans, lossy scalar
conversion and trainable tensors. They are neither preferences nor search budgets.
There was no existing shared hard-limit record for these three fields; the
existing estimate records costs rather than ceilings.

Services owns the new private `CheckedExecution` view and `inspect_execution`.
It composes existing APIs and adds no planner, compiler, numerical kernel,
serializer or mutable registry. Compiler and Runtime implementation paths remain
unchanged apart from the pure Compiler limit record. Existing `fq.compile`
returns CircuitIR; ExecutionOptions, ExecutionResult, public exports, snapshots,
dependencies and CI configuration are unchanged.

The check reads the existing sealed payload and the actual instruction layers.
It checks layer order/qubit summaries, analysis dimensions, RuntimeConfig's
existing manifest against the sealed backend/device/precision, raw execution
controls and canonical program identity. The full plan identity already binds
resolved options, compiler and environment; no second fingerprint protocol is
introduced. Returned resource estimates are freshly derived, so editing a
returned operation-count dictionary cannot change the checked evidence.

`fq.run(checked.plan)` uses the existing no-replanning Runtime path. Executing
the same source through ordinary `fq.run(source, options=...)` also verifies when
the resulting executable/configuration agrees. Inspection, resource access and
result verification never execute, compile or replan. Tests disable those calls
to enforce the boundary.

The existing plan serializer retains routing in its canonical program but omits
the optional raw routing diagnostic on restoration. A recorded-plan view uses
that canonical witness when the optional diagnostic is absent; it does not patch
the plan. When diagnostics are present, conflicting controls are rejected.

## Admitted routing and refusal boundary

Only exact local CPU torch/pytorch statevectors, one batch, static registered
unitaries and a full state without observations/measurements/noise are admitted.
Result checks preserve dtype, precision and dimensions. They verify association
with a plan, not authenticity of manually forged tensor data or physical state
preparation.

Unrouted plans use identity layout. Routed plans initially admit only the known
v1 undirected persistent-layout record, identity initial placement, no additional
idle physical slots, an explicitly matching CouplingMap and intact routed SWAP
witnesses. The checker reuses Compiler's witness parser and Layout replay, checks
both pre-restore and final permutations, verifies actual two-qubit locality, and
projects bounded disjoint named logical ports through that verified layout.

Missing route metadata, inconsistent counts/permutations, unknown strategies or
schemas, mismatched topology and conflicting raw controls are refused. A
post-routing basis conversion that replaces witness SWAPs with cx instructions
is also refused: the circuit may still be numerically right, but this slice does
not yet have a checked transformation chain carrying that layout proof. It does
not infer correctness from `mapping_restored=True` or silently admit directed or
allocated routes. These are explicit future integration cases, not fake support.

## Evidence and handoff

Conformance tests use the same consumer with live Runtime plans and a test-only
recorded-plan replay fake based on the existing serializer. The replay fake is
transport conformance, not independent physics evidence. Real CPU tests separately
verify direct checked-plan execution, ordinary-source execution, plan/source JSON
round trips, nonidentity named-port probability projection and complete relative
phase. Both CS-05 arithmetic schedules are checked with phased preparation and
clean workspace in the same full executable scope, on at most ten qubits.

Negative cases cover same-width different programs, changed options and raw
configuration, corrupt layers/dimensions, stale target/limit claims, missing
plans, mismatched result shape/precision, invalid named ports, incomplete SWAP
witnesses and supported-name/unsupported-conversion distinctions. Exceeded
resource diagnostics list every blocked ceiling, and a real budget retry is
executed. This is not formal verification of arbitrary caller-authored sources.

B must bind source/candidate identities, preparation, workspace allocation
history and target transformation records to the checked executable. CoreIR
alone does not certify workspace ownership or cleanup, so A accepts no unchecked
workspace count or unenforced max_workspace field. B also owns deterministic
selection, budget exhaustion, complete candidate tables and verified recovery
suggestions. C replaces the frozen CS-04 admission guard and completes ordinary
named reading and the user experience; this A slice does not relax that reader.

No independent human usability trial or general Qmod compatibility is claimed.
See the [private contract and handoff table](../api-changes/FQ-SYNTH-CS06-A-20261029.md)
before extending the scope or publishing an API.

## Validation on 2026-10-29

- 59 new cases passed: 42 unit conformance/refusal cases and 17 real CPU
  integration cases. The example is exercised as its documented subprocess.
- 668 distinct focused regressions passed: 628 construction/arithmetic,
  Services, examples, result/API and vocabulary cases; 27 execution-plan cases;
  and 13 Hybrid IR/verifier cases. The 59 new cases are included in the 628.
- The two new modules cover 176/179 statements and 81/84 branches, 97.72%
  combined coverage (rounded to 98%). This is not whole-package coverage.
- Whole-package strict mypy passed all 863 source files. The example also
  passed strict mypy. Ruff, Black, architecture, dependency policy, public API
  baseline, construction acceptance, vocabulary, ownership scope, language,
  documentation links and diff checks passed.
- Local `python -m pytest -m 'smoke or unit' -q --maxfail=3` stopped after
  4262 passed, 121 skipped and three failures in 197.55 seconds. These are the
  previously recorded macOS native QFT fusion-count assertions in
  `tests/team/simulation/test_controlled_phase_graph.py`: two dtype cases of
  `test_qft_hadamard_phase_fusion_matches_graph_path_and_reduces_passes` and
  `test_product_state_qft_uses_hadamard_phase_fusion`, each expecting 28 passes
  and observing 44 in this environment without the native CPU extension.
  Simulation and these assertions are unchanged. The local full default tier
  is not green; fresh Linux CI is required rather than reusing CS05 results.

The reported qubit/operation/depth values are static costs of the actual full
executable. They do not measure hardware latency, fidelity or peak memory.
