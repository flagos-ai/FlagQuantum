# One sampled Ising pair-charge trajectory

The user explicitly approved the five decisions in
[proposal 091](../development/API_CHANGE_PROPOSAL_091_ISING_SAMPLED_OBSERVATION.md)
on 2026-10-24. The proposal document #818, replaceable baseline #821, Core #824
and Simulation #827 merged under their respective authorizations. Runtime is
delivered separately by #838. Every PR still requires its own merge grant;
contract approval is not blanket merge permission.

## Shared baseline and replacement

The baseline established four source/planning/execution/JSON factories and
independent consumers before production implementation. Its disposable reference
composed existing validation, tensor encoding, exchange/Kraus actions, terminal
sampling and ExecutionResult/MeasurementResult. It used only a
`test.ising_sampled_recipe`, never an ExecutionPlan or a production source
schema, and attached no fake plan.

The four factories now use real Core construction, `fq.plan`, `fq.run` and
`ExecutionPlan.from_json`. The obsolete reference is deleted. Independent
consumer/oracle function ASTs are unchanged: fixed F/R matrices, both initial
channels, noncommuting suffixes, zero-weight contributions, full conditional
densities, trace, purity and weighted unread equality retain atol=1e-12, rtol=0.
Current pair IDs, frozen source/output snapshots, name-before-kind resolution,
private RNG and exact seeded direct/planned/JSON replay retain their assertions.
The shared 4,096 single-index draws retain the predeclared 1,856..2,240 psi range;
no build-specific outcome sequence is encoded.

The existing execution-plan candidate now marks the approved extension
`implemented`, with `production_implemented=true`. Its exact guard still
rejects unapproved mutations. This flag does not promote a capability level.

## Domain responsibilities

Core reuses ProgramRecord/Operation and the existing physical verifier/codec.
The fluent zero-argument `measure_pair()` records literal null physical data;
a shared atomic append guard permits one sampled or unread observation total.
Braid-only source 1.1 and unread source 1.2 bytes remain unchanged. Core owns no
RNG, execution state or numerical observation algorithm.

Simulation reuses its ordered exchange kernel, `SuperOperator.from_kraus` and
existing density validator. It computes the prefix and Runtime-selected
conditional continuation, normalizing by the original positive branch trace.
Invalid indices and zero-weight branches raise ExecutionError without division.
The full-density entry rejects sampled sources before exchange arithmetic; it
cannot silently turn a selected trajectory into an unread mixture.

Runtime extends the existing native executor and plan identity/preflight. Only
the sampled operation selects plan 1.3 and environment protocol 1.1 with the
exact `rng_torch_version`. Existing terminal probability checks preserve
original weights beside corrected draw probabilities; the existing private CPU
generator/draw selects one outcome. Simulation continues that branch. Reading
or executing an incompatible sealed RNG environment fails before a kernel/draw.
No public signature, result field, IR, registry, controller or solver is added.

Requested expectation records come first, followed by one existing
`mid_circuit_measurement` record with empty qubits, int64 index `(1,1)`, original
probabilities, correction, source operation index and current pair IDs. The
conditional state retains shape `(1,2,2)`. No terminal samples, classical bit
register, event time or feedback is inferred.

## Runnable public workflow

```python
import torch
import flagquantum as fq
from flagquantum.experimental.simulation import IsingBraidProgram
program = IsingBraidProgram().exchange(1).measure_pair().exchange(0)
plan = fq.plan(program, options=fq.ExecutionOptions(shots=1, seed=17),
               outputs=fq.expectation(program.projector("psi"), name="psi"))
result = fq.run(fq.ExecutionPlan.from_json(plan.to_json()))
observed = result.measurement("mid_circuit_measurement")
index = int(observed.value[0, 0])
print(plan.schema_version)
print(observed.metadata["outcome_labels"][index] in ("1", "psi"))
print(abs(float(result.expectation("psi")[0]) - index) < 1e-12)
print(torch.equal(result.state, fq.run(plan).state))
```

Executed output: `1.3`, `True`, `True`, `True`. No particular label is promised
across Torch builds; the sealed plan repeats exactly in the reviewed environment.

Run the real consumers and public integration workflow with:

```bash
python -m pytest tests/unit/test_ising_sampled_contract.py tests/unit/test_ising_sampled_runtime.py tests/integration/test_ising_sampled_execution.py -q
```

## Compatibility, resources and limits

Compatibility tests retain eight original unread source/plan 1.2 captures from
before implementation. Existing 1.1, circuit, Rydberg and Lindblad fixture
authorities remain unchanged. A further pre-Runtime capture preserves seventeen
complete old plan JSON strings and ten native sources for exact comparison.
Strict rehashed tests cover source/plan mismatches, old environment refusals,
explicit sampling options and RNG build requirements; the old environment 1.0
mapping and old bytes are not rewritten.

Only one CPU complex128 trajectory in the fixed four-sigma total-vacuum sector
is admitted. `shots=1` and a Python integer seed in `[0,2**64-1]` must be explicit.
The inherited-control, 215/216-byte and dense-ceiling tests refuse unsupported
requests before prefix evolution. The sampling input/probability/sample subtotal
is 216 bytes, not a process-RSS bound; old state storage remains 64 bytes.

Gradients, multi-shot native batching, multiple observations, feedback, atom-loss
inference, noise, alternate backends/devices, approximate fallback, cost
calibration, Remote, physical hardware and other anyon sectors remain outside
this delivery. This bounded path does not certify full M3/M4.
