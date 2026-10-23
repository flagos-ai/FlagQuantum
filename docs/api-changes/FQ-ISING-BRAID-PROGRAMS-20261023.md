# Ising braid programs: owner approval and replaceable consumers

## Owner decision and delivery order

On 2026-10-23 the owner explicitly replied "Approve the contract and conditional
merge of the documentation" (English rendering) to the five-decision request
for [proposal 085](../development/API_CHANGE_PROPOSAL_085_ISING_BRAID_PROGRAMS.md).
This approves the fixed four-sigma, total-vacuum frontend, shared root
planning/execution with optional circuit analysis, native-only source and plan
1.1, existing operator/terminal-instrument reuse, and baseline-first delivery.

The owner separately authorized normal merge of documentation PR #758 only
after its latest applicable checks, complete CI/GPU workflows and repository
rules pass. Later PRs require separate merge authorization. Document merge alone
does not supply implementation approval; the explicit owner decision does.

This consumer baseline is stacked on documentation head
`3dac28c30a437269a8d39f476d33b3a28da5784d`. It is the first Integration step,
not Core implementation, interface admission or Runtime activation. Existing
protected candidates, namespaces, ProgramRecord, plan readers and public
snapshots remain unchanged. Subsequent contract-admission work must extend
their existing authorities before domain activation; it cannot waive them.

## Replacement seams and verified scope

The four pytest fixtures in `tests/unit/test_ising_braid_contract.py` are
program_factory, braid_planner, braid_runner and plan_reader. They currently
bind to small test-local mathematical references. Replace them with the admitted
frontend, root planner/executor and ExecutionPlan reader without changing the
consumer assertions or physical/statistical budgets.

The reference source stores only the initial channel and ordered exchanges;
it is not a new Core IR. The reference plan captures only this fixture and
cloned output matrices through the existing Core scalar/tensor encoding and
canonical JSON/hash helpers. It is **not** an executable-plan 1.1 producer,
source verifier, general input validator or public serialization implementation.
The reference runner delegates density action to SuperOperator and results to
the existing ExecutionResult/MeasurementResult. It attaches no fabricated public
plan. Named expectations reuse the existing output-name/index metadata; no
second result selector is introduced.

The 23 consumers cover both allowed initial channels, empty and ordered braid
words, clockwise adjoints, full complex density, result/output axes, direct
versus planned execution, a single matrix output descriptor, JSON numeric replay, frozen source and matrix snapshots, source identities for distinct words,
both adjacent braid relations, distant commutation, inverse words, and explicit
discrete-gradient/Pauli-output refusals. Independent consumers derive the central
generator as F-adjoint R F; the reference uses its separate closed form.

Terminal charge observations use the real admitted sample_terminal entry,
ordered complete QND projectors, same-environment seeded replay and global RNG
isolation. The unread map removes relative coherence while preserving charge
probabilities. The density/probability budget remains absolute 1e-12 with zero
relative tolerance; shot frequencies use a separate six-binomial-standard-error
plus 2/shots bound. This does not simulate destructive fusion or anyon removal.

These consumers do not yet establish actual Core source types, full 1.1 schema
validation, environment/resource preflight, plan attachment, output admission,
worldline inspection, all malformed-input errors, device support, or successful
compiler/Deployment/Remote/Lindblad refusals. Those remain the approved
Integration/domain gates. In particular, the expanded physical-context union
must not reach Rydberg-only field assumptions in Lindblad. Generator-level
unitarity/global-phase checks remain separate from density-map equivalence.

## Compatibility evidence captured before implementation

`tests/fixtures/ising_braid_legacy_plans.json` captures two actual circuit
execution plans 1.0 and one actual Rydberg Lindblad plan 1.7 with physical source
1.0 from the stack base above. Tests reconstruct them through the current APIs
and compare complete canonical JSON bytes, then read and write them again.
Do not regenerate these records to accommodate the new native profile.

Existing Lindblad 1.3/1.4/1.5/1.6 golden fixtures and the pre-081 ordinary Program
record/hash consumers remain authoritative and are reused unchanged. This
baseline adds the missing executable-circuit and native-1.7 captures rather than
copying those fixture families or adding another contract registry.

## Runnable user reference

The existing user workflow remains executable without the proposed frontend:

```bash
python -m examples.constrained_basis_reference
python -m pytest tests/unit/test_ising_braid_contract.py -q
```

The first command prints one-excitation transfer 0.50000000, control derivative
1.00000000, fusion-channel probability 0.50000000, and true braid-relation/basis
label checks. It remains a caller-owned matrix reference. The second currently
passes all 23 consumer cases; its fake seams do not establish public native
support. The production frontend example in proposal 085 becomes runnable only
after the remaining admitted implementation is delivered.

## Validation record

All 23 focused cases passed on Python 3.10, 3.11 and 3.12 after correcting the
test-local reference's output-name metadata to use the existing Runtime keys.
The first reference run failed 13 named-expectation consumers because it supplied
only a measurement name; neither consumers nor numerical budgets were weakened.
The existing-plan/matrix compatibility selection passed 70 cases. All 11
repository checks passed, as did floor/current Black and Ruff checks and strict
mypy for this test file. The CI-aligned strict package check passed all 786
source files using `--python-version 3.12`. An initial package check omitted
that CI target override and reported four errors in unchanged optional JAX/NumPy
paths; the command was corrected, without source edits or suppressions.
The complete default tier passed 14,886 cases with 167 original skips in
304.27 seconds. Remote verification remains a separate acceptance gate.
