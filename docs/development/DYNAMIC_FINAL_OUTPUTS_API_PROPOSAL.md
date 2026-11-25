# Final samples and counts from local dynamic circuits

Status: approved contract. Implementation and acceptance follow in this PR.
Proposed title: `feat(runtime): support final outputs for local dynamic circuits`.
Owner: Runtime, with integration review for the public behavior contract.
Classification: `single_device_fast_path`.

## User problem and proposed behavior

Existing local dynamic execution accepts `fq.run(circuit, shots=N)` and
produces final samples plus classical-register data. Its planner currently
rejects every explicit `outputs` request. Admit existing
`fq.samples()` and `fq.counts()` requests for a bounded dynamic execution
domain. This is an additive behavior change under API-owner review.
No new public function, output kind, options field or circuit type is needed.

First domain: existing local CPU PyTorch dynamic trajectory execution,
one circuit batch, positive shots, default initial state, and the already
supported noiseless dynamic instruction set. An explicit request may
contain one full-register computational-basis sample request, one counts
request, or one of each, in caller order. Names follow existing output
selection. Shots follow the existing facade resolution. Both requests in
one run use identical shots and the same final rows.

Use logical qubit order `(0, ..., n-1)` for this slice. Returned samples
retain the existing dynamic `(shots, n_qubits)` shape for batch size one.
Counts use the existing list-per-batch, binary-string-key convention.
Preserve the existing `result.samples` access and
`result.measurement("mid_circuit_measurement")` register record, including
when only counts are requested. The register record represents classical
slots, so it does not promise history when a slot is overwritten.

A program with an explicit terminal measurement block remains outside
this additive domain. Reject it with a message explaining that this
request currently uses the existing final readout. This prevents accepting
a workflow in which the explicit terminal block and the engine's final
readout would both occur. Calls without `outputs` preserve their existing
behavior. Subsets, custom wire order, Pauli-basis outputs, state and
expectation requests, noise, gradients, multi-batch execution, hardware
feedback, and remote execution remain outside the new admission.

## User workflow after implementation

```python
from collections import Counter

import flagquantum as fq
import torch
from flagquantum.dynamic import DynamicCircuit

circuit = (
    DynamicCircuit(2)
    .h(0)
    .measure(0, classical_bit=0)
    .conditional("x", 1, classical_bit=0)
)
result = fq.run(
    circuit,
    outputs=(fq.samples(name="final"), fq.counts()),
    options=fq.ExecutionOptions(seed=7),
    shots=100,
)

mid = result.measurement("mid_circuit_measurement").value
final = result.measurement("final").value
counts = result.counts[0]

assert torch.equal(final, result.samples)
assert torch.equal(final[:, 0], mid[:, 0])
assert torch.equal(final[:, 1], mid[:, 0])
assert counts == dict(Counter("".join(map(str, row)) for row in final.tolist()))
assert sum(counts.values()) == 100
print("mid", tuple(mid.shape))
print("final", tuple(final.shape))
print("counts total", sum(counts.values()))
```

Expected stable stdout, independent of exact RNG counts:

```text
mid (100, 1)
final (100, 2)
counts total 100
```

A caller can request either result alone:
`fq.run(circuit, outputs=fq.samples(), shots=100).samples` or
`fq.run(circuit, outputs=fq.counts(), shots=100).counts[0]`.
The example declares the intermediate measurement and feedback. The engine
performs its existing terminal readout once. The published example must be
run on the final reviewed head outside the checkout.

## Implementation boundary

1. At dynamic admission, validate the bounded subset of existing lowered
   MeasurementNode requests. Reuse their shot count with the existing
   ExecutionOptions resolver, including direct `fq.run(..., shots=N)`.
   Reject inconsistent lowered requests. Carry the requested kinds, order,
   names and shots through the existing IR and ExecutionPlan v1 payload.
   Keep source instructions and their measurement/feedback dependencies
   intact. Recheck admission and shot agreement on plan replay.
2. Keep `run_dynamic` as the one execution and sampling call. Adapt its
   final sample tensor at the plan execution boundary. A sample request
   selects that tensor and labels the requested output. A counts request
   uses the existing counts-from-samples helper on the same rows, adding
   the batch axis only for histogram construction. Preserve the existing
   mid-circuit result. A named sample record appears once in the output
   list rather than duplicating the default final record.
3. Keep the existing default dynamic projection and internal select
   lowering behavior. Preserve static output lowering, remote preflight,
   the result field schema, and all existing thresholds. No separate
   sampling operation is introduced by output projection.

The planner is already close to its size ceiling. Put focused admission,
shot resolution and final-row projection in one private Runtime module.
No new sampler, callback registry or public execution configuration.

## How to prove that no extra measurement was added

- At the same seed, compare legacy dynamic execution, sample-only,
  counts-only, and one call requesting both. Verify the final rows and
  mid-circuit data agree and the two requested outputs are the same
  batch's records and their independent histogram.
- Instrument the actual execution boundary to assert one call to
  `run_dynamic` per request. Compare terminal draw counts against the
  legacy baseline separately for trajectory and batched strategies. The
  trajectory witness observes the existing final sampler. The batched
  witness observes the existing batched terminal draw. Include an extra
  unmeasured superposition qubit so bit equality alone cannot hide another
  draw of an already collapsed final state. Compare each strategy against
  its own seeded baseline.
- Assert that the lowered output request adds no measurement instruction
  and changes neither the intermediate measurement count nor the
  feedback behavior. Compare counts with an independent histogram in
  logical qubit order, including asymmetric final rows to catch reversal.
- Reject an explicit terminal measurement block before executor entry.
  Prove unsupported basis, subsets, result kinds, invalid shot counts
  and conflicting predeclared requests fail before any new run. No
  manual sampling is permitted on an admission failure.
- Serialize and reload a plan with both existing output kinds, run it
  through the stable plan entry, and verify names, shots, register data
  and counts. Confirm malformed or inconsistent replay data is refused.
  Run existing static and dynamic no-outputs regressions unchanged.

Read-only preflight: the current H/measure/conditional-X example returned
an intermediate register of shape (8,1) and final samples of shape (8,2).
Both final columns equalled the measured classical bit. Explicit
`fq.samples()` and `fq.counts()` requests were each rejected by the
current planner. The new acceptance is planned and has not run.

## Exact path scope

1. `flagquantum/runtime/planner/__init__.py`
2. `flagquantum/runtime/plan_execution.py`
3. `flagquantum/runtime/dynamic/_final_outputs.py` (new private helper)
4. `contracts/dynamic-final-outputs-v1.json` (new behavior contract)
5. `docs/development/DYNAMIC_FINAL_OUTPUTS_API_PROPOSAL.md` (new)
6. `tests/unit/test_dynamic_final_outputs_contract.py` (new)
7. `tests/integration/test_dynamic_final_outputs.py` (new)
8. `examples/local/dynamic_final_outputs.py` (new)
9. `examples/README.md`
10. `docs/guides/LOCAL_WORKFLOWS.md`

Contract, fake and conformance first, then implementation in the same
direct-main PR. Run ownership preflight against this exact list before edits.
This work is independent of PR #1096 and the terminal QEC converter. No
changes to Circuit builders, dynamic sampler, release schemas, dependencies,
CI/runner settings, or historical evidence. No research execution.

## Verification commands

```bash
python tools/check_team_scope.py --team integration --files <the-10-listed-paths>
python -m pytest -q \
  tests/unit/test_dynamic_final_outputs_contract.py \
  tests/integration/test_dynamic_final_outputs.py \
  tests/test_dynamic_circuit.py \
  tests/unit/test_execution_options.py \
  tests/unit/test_dynamic_terminal_measurement_normalization.py \
  tests/test_algorithm_examples.py::test_examples_readme_names_every_example_that_does_not_use_the_root_alias
python -m examples.local.dynamic_final_outputs
python tools/check_architecture.py
python tools/check_dependency_policy.py
python tools/check_docs_links.py
python tools/docs_source_of_truth.py --check
```

Run Black, minimum Ruff and strict Mypy on the changed Python files, plus
language, hygiene and diff checks. Confirm new unit and integration tests
are selected by the existing tiers. Retain final-head stdout and the
existing installed acceptance when the implementation is approved.
