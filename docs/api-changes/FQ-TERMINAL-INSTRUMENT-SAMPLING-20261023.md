# Terminal instrument sampling authorization and contract baseline

## Owner decision

On 2026-10-23 the owner explicitly replied "Approve 083; proceed with the
contract baseline first" (English rendering) to the five-decision implementation
request for [proposal 083](../development/API_CHANGE_PROPOSAL_083_TERMINAL_INSTRUMENT_SAMPLING.md).
This approves all five decisions and their staged delivery order. Document merge
#716, earlier native evolution approval and the training/view merge grants did
not supply this authorization. No blanket PR merge permission is granted.

The approved entry is Runtime-owned sample_terminal with ordered tensor Kraus
branches, CPU terminal density matrices and the existing MeasurementResult.
Existing SuperOperator, density checks, the private probability draw, Core tensor
encoding and Torch generators remain the implementation owners. There is no
second IR, result envelope, parser framework, registry, solver or trainer.
Existing Lindblad shots/seed refusals and plan/CircuitIR versions stay unchanged.

## Replaceable baseline

The inspected integration base is main
`71ef9a57258babe0506c9ba0ae99a76923e0d60f`. The baseline adds no production
implementation, Runtime export, API candidate, public snapshot or new plan schema.

The three test-local fixture seams in
`tests/unit/test_terminal_instrument_contract.py` are terminal_sampler,
presence_instrument and fock_instrument. The branch builders are bounded
mathematical inputs, not proposed public model factories. The sampler reference
accepts only valid bounded fixtures: it delegates branch action to SuperOperator,
draws to Torch, capture to the existing Core codec and output to MeasurementResult.
It is not an input validator, capability checker or Runtime dispatch path.

The consumers exercise both complex dtypes; efficiencies 0/0.8/1; ground,
Rydberg, lost, mixed and coherent inputs; ordered pair flags; finite-Fock QND
statistics and unread dephasing, including a complex basis change that forbids a
diagonal-only sampling shortcut; distinct instruments sharing effects; B=1/B=2;
zero-probability outcomes; fixed/default seeds; capture, JSON-compatible metadata
and global RNG isolation. Physical budgets remain 1e-10 and 2e-5. Empirical draws
use the separately declared six-binomial-standard-error plus 2/shots bound.
The existing qubit density sampler retains independent seeded index/bit consumers.

Replace terminal_sampler with the admitted Runtime entry without changing those
consumer assertions. Keep branch construction explicit; this baseline does not
authorize a new public model-factory API. Add production refusal/resource tests
with the implementation, rather than building a second request validator in the
fake. In particular, passing this baseline does not establish invalid-input,
device, gradient, byte-ceiling or no-RNG-advance-on-refusal behavior.

The existing old-plan/source fixtures are reused unchanged. Their current
consumers, old circuit measurement consumers and full applicable verification
remain required for subsequent factoring and implementation. Historical private
source hashes must not become permanent implementation locks.

## Runnable current-API reference

```bash
python -m examples.terminal_instrument_reference
python -m pytest tests/unit/test_terminal_instrument_contract.py -q
```

The example continues to use current SuperOperator maps and illustrative Torch
draws; it does not call an unavailable framework entry. Its probability output is
presence [0.68, 0.32] and Fock [0.6, 0.3, 0.1], with reproducible seeded draws,
no-flag conditional ground 0.88235294 and a changed unread coherence. The final
line states "Framework terminal instrument API: approved, not implemented".

## Baseline integration evidence

PR #726 landed as `81f817efe234c35dee47cf7e6b002c543655f51a`.
Its final head `cdcb8d7017cd90e4db5ce22b5dab9c6c484602f9` passed all
41 applicable checks; two manual GPU-pool checks retained their existing skips.
The complete [CI run](https://github.com/flagos-ai/FlagQuantum/actions/runs/39889495855)
and [GPU run](https://github.com/flagos-ai/FlagQuantum/actions/runs/39889495864)
succeeded before merge. The resulting
[main smoke](https://github.com/flagos-ai/FlagQuantum/actions/runs/39892408347)
also succeeded. GitHub recorded the merge externally; this record does not
reinterpret implementation approval as an agent merge grant.

## Runtime interface admission

The next integration change extends the existing
`contracts/execution-result-v1-candidate.json` and
`tools/public_api_snapshot.py`; it creates no separate manifest or checker.
The admission records the approved Runtime-only signature, retained Runtime
export prefix and exact result metadata fields. Metadata object key order is
not a new promise. Existing MeasurementResult fields and schema version remain
protected by the same checker; old snapshots and plan fixtures stay unchanged.

The candidate's implementation_required value records the delivery stage, not a
user-facing option. While false, the checker requires the existing Runtime
exports and refuses even a non-exported placeholder attribute. The real Runtime
implementation must change it to true in the same delivery: thereafter the
checker requires the appended export and exact signature, and removal fails.
Runtime owns that transition after the admission lands. Returning it to false
to conceal removal requires a new explicit contract decision. Neither admission
nor a passing signature inspection establishes the sampling semantics.

Admission tests exercise both stages using isolated namespace mutations and
verify the actual namespace against the recorded current stage. The numerical
baseline remains unchanged. The real implementation must replace only its
terminal_sampler fixture and satisfy the same numerical consumers plus the
production refusal/resource tests. Until then the runnable reference below
continues to report that the framework API is not implemented.

## Remaining delivery gates

1. The replaceable baseline has landed with the evidence above; retain its
   independent consumers through the remaining stages.
2. Admit the approved Runtime export/metadata surface in a separate integration
   change. Preserve MeasurementResult fields, schema_version and existing export
   prefixes; do not regenerate unrelated snapshots or old fixtures.
3. Implement Runtime consumption behind those approved boundaries, reusing the
   existing numerical owners and shared private draw. Replace the fixture and
   preserve all independent consumers. Add strict type/value/completeness,
   device/gradient/byte-ceiling refusals and their pre-draw evidence.
4. Run supported Python minors, the appropriate testing tiers, old measurement
   and plan compatibility consumers, repository gates and full CI/GPU. Every PR
   includes runnable user code and expected results and requires its own merge
   authorization. Baseline approval or merge is not blanket merge permission.

Terminal samples are not hidden loss truth or timestamped detector events.
Feedback, QEC decoder records, motional heating and concrete hardware/service
submission remain later gates. Neither the presence reference nor finite Fock
counting certifies native photonic hardware, a destructive photodetector or M3
completion.
