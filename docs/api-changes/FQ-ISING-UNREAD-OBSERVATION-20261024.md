# Ising unread pair-charge observation

Proposal: [089](../development/API_CHANGE_PROPOSAL_089_ISING_UNREAD_OBSERVATION.md).
The five decisions were explicitly approved on 2026-10-24. Document PR #793
merged as `1570fb9f74efa85878255ca6f94cb0063b0abff8` after its full applicable
CI/GPU succeeded. This record grants no additional PR merge permissions.

## Shared baseline

The [existing execution-plan authority](../../contracts/execution-plan-v1-candidate.json)
records the approved narrow extension. Its live braid-only version remains 1.1.
Production source/plan 1.2 and `dephase_pair()` are **not implemented** by this
baseline. Existing stable root signatures and result fields are unchanged.

[Independent numerical consumers](../../tests/unit/test_ising_unread_contract.py)
cover both initial fusion channels, zero-weight branches, ordered exchange/channel
composition, full complex density, trace and mixed-state echo, named expectations,
frozen source/output snapshots, deterministic recipe replay and untouched private
RNG state. They use fixed F/R references at absolute tolerance 1e-12 and rtol=0.
The four replaceable fixtures are source construction, planning, execution and
reading. Later implementation replaces these factories; the consumer assertions
and mathematical oracle must remain unchanged.

The test-only reference composes existing Simulation exchange actions and
`SuperOperator.from_kraus`; it reuses existing tensor encoding, canonical JSON/hash
and `ExecutionResult`/`MeasurementResult`. Its `test.ising_unread_recipe` is a
private fixture recipe, **not** a production native source or executable plan.
It does not attach a fabricated public plan to results. It certifies numerical
consumers and snapshots, not source/schema, environment or resource admission.

[Compatibility consumers](../../tests/unit/test_ising_unread_compatibility.py)
read six exact braid-only source/plan captures made before implementation at
`4be2375eff47c112ce8d8e24f8206ed5592130d8`. The archived JSON strings and SHA256
values were copied without regeneration. Existing circuit, Rydberg and Lindblad
capture authorities remain unchanged and keep their existing tests.

## Remaining domain deliveries

1. Core adds the bounded builder operation and strict source 1.2 verification/
   codec in the existing ProgramRecord. Test current-pair worldline semantics,
   argument and duplicate refusals, declared null only for the new opcode, and
   refusal of malformed, extra or inconsistent source data.
2. Simulation extends the existing ordered action kernel with the fixed unread
   channel. It adds no solver, physical time or result wrapper.
3. Runtime admits matching native plan/source 1.2 through existing planning,
   identity, preflight, replay and result authorities. Replace all four baseline
   factories with the real path. Test rehashed inconsistent versions, allocation/
   environment and options refusals, exact public replay and old bytes.

Each PR needs separate user merge authorization and latest complete applicable
CI/GPU. These deliveries do not admit sampled branches, feedback, classical bits,
timestamps, Remote, hardware or another anyon sector.

## Runnable workflow supported today

The proposal's current-API mathematical reference remains the executable user
example. Run its code to obtain `0.5`, `0.0`, `True`: an unread first-pair channel
between an exchange and its inverse destroys the coherence needed for the echo,
leaving I/2. It is caller composition, not implemented native observation.

Run the baseline consumers and existing Ising consumers with:

```bash
python -m pytest tests/unit/test_ising_unread_contract.py tests/unit/test_ising_unread_compatibility.py tests/unit/test_ising_braid_contract.py -q
```
