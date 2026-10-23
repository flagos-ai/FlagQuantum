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
statistics and unread dephasing; distinct instruments sharing effects; B=1/B=2;
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

## Remaining delivery gates

1. Verify and land this baseline only after its own explicit merge grant and
   latest complete applicable CI/GPU success under repository rules.
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
