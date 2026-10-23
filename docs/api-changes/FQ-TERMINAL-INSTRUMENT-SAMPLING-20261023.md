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
draws; the separate sampling example below calls the admitted Runtime entry. Its probability output is
presence [0.68, 0.32] and Fock [0.6, 0.3, 0.1], with reproducible seeded draws,
no-flag conditional ground 0.88235294 and a changed unread coherence. The final
line identifies "Framework terminal instrument API: flagquantum.runtime.sample_terminal".

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

The separate integration change #730 extended the existing
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
production refusal/resource tests. The numerical reference remains available beside the real Runtime sampling example.

## Runtime implementation and remaining delivery gates

Admission PR #730 was normally merged with explicit conditional owner
permission as `ff69b2526809772b7fadde5bcb420f9f5899dcf6`. Its final head
`ae4fe416fe74d78f2dd3112ba9cf819b9835eeda` passed 41 applicable checks and the
two original manual GPU-pool skips. Complete
[CI](https://github.com/flagos-ai/FlagQuantum/actions/runs/39894277586),
[GPU](https://github.com/flagos-ai/FlagQuantum/actions/runs/39894277587) and
[merged-main smoke](https://github.com/flagos-ai/FlagQuantum/actions/runs/39897061398)
succeeded. Live repository protections and rules were unchanged by the merge.

The Runtime delivery activates the admitted entry and binds terminal_sampler to
it, retaining all numerical consumer assertions. It delegates density checks to
the existing Lindblad helper, each branch action to SuperOperator, capture to
the Core tensor codec, and drawing to a private primitive shared with the old
qubit density adapter. The old adapter retains its normalization, shapes and
generator policy, including unseeded global draws. New terminal calls always
use a private CPU generator. KrausChannel remains qubit-only; it is not adapted
to change its existing power-of-two or tolerance semantics.

The sum of state, Kraus, branch-probability and int64 output numeric storage is
checked against the existing dense ceiling before capture or drawing. Gradients
and non-CPU tensors are refused before conversion. Invalid inputs never create
a generator, capture matrices or draw. Metadata retains original checked branch
weights beside the maximum roundoff correction; missing outcomes are rejected.
A real native-plan integration also feeds the existing 1.7 moving-pair density
into the same sampler, including a batch member with both atoms lost. At
efficiency 0.8 that member still has no-flag probability 0.04, distinguishing
observed flags from physical survival; sampling leaves the plan JSON unchanged.
The document-code guard reads the existing admission to allow only the approved
Runtime entry, retaining refusals for other Runtime imports.

The implementation adds no result fields, public model builders, IR fields,
plan versions, event records or feedback.

Run the actual user path:

```bash
python -m examples.terminal_instrument_sampling
python -m pytest tests/unit/test_terminal_instrument_contract.py tests/unit/test_terminal_instruments.py -q
```

Expected probabilities are presence [0.68, 0.32] and Fock [0.6, 0.3, 0.1].
Each result has 4096 outcome indices, private-seed replay is true, and global
Torch RNG state is unchanged. See the complete output and copy-ready code in
[the examples guide](../../examples/README.md#terminal-instrument-mathematical-references).

The Runtime implementation PR must finish supported Python-minor validation,
the appropriate testing tiers, old measurement/plan consumers, repository gates
and latest complete CI/GPU. It requires its own explicit merge grant; #730's
permission does not cover it. The baseline, admission and implementation each
retain their separate verification evidence.

Terminal samples are not hidden loss truth or timestamped detector events.
Feedback, QEC decoder records, motional heating and concrete hardware/service
submission remain later gates. Neither the presence reference nor finite Fock
counting certifies native photonic hardware, a destructive photodetector or M3
completion.
