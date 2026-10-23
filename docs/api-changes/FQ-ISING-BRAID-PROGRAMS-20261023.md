# Ising braid programs: owner approval, replaceable consumers and Core source

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

## Core source admission after the baseline

Baseline #761 merged normally as
`3b02eabea87d8474885086bf46f5d89efcd956ed` after its latest complete CI/GPU and
all applicable checks succeeded. The Core source delivery starts from main
`6ab01ab245067358b2b62359345d45640f1c4b6f`. The earlier fake limitations above
describe that original baseline; the new source tests now exercise real Core
types, without claiming the remaining execution replacement gate is complete.

`core/ising.py` owns IsingBraidProgram construction and fresh CPU charge
projectors. The existing physical-profile module owns frozen IsingContext and
IsingExchange records and validates their fixed sector/tree/gauge and ordered
word. ProgramRecord and Operation gain no fields; their existing typed unions
admit these values. Append order is never sorted or converted to numerical time.
The existing source codec emits 1.1 only for Ising and keeps Rydberg at 1.0.
No operator registry, parameter binder, solver or second IR is introduced.

The approved internal candidate evolves from 0.4 to 0.5, listing both profiles
under physical_profiles. Its existing strict reader checks the exact reviewed
declarations and resolves their real types/frontends. Prior approvals,
authorities, diagnostics, SSA kinds and Quantum/Target refusals remain intact.
Only experimental.simulation gains the new frontend; Stable Core exports,
CircuitIR 1.0, executable circuit plan 1.0 and Lindblad 1.3-1.7 are unchanged.

Core remains the primary implementation domain. The narrow Simulation/Lindblad
consumer admission required by proposal 085 is included before exposing the
frontend: its high-level facade rejects discrete sources after Core validation
and before Rydberg attribute access; its old plan reader refuses an Ising source
even after a caller recomputes the old plan hash. These edits do not change model
assembly, bindings, Liouvillians, integrators, adjoints or numerical plans.
Existing generic Core guards also refuse circuits, compilation, OpenQASM,
Deployment and Remote without target resolution or external side effects.

Native root planning/execution is still refused in this source stage. Runtime's
approved source/plan admission will replace only these temporary root-entry
refusal assertions; circuit, Deployment, Remote and Lindblad refusals remain.
Executable plan 1.1, optional circuit analysis, exact Simulation F/R actions,
native results and terminal sampling replacement remain subsequent deliveries.
Neither this source nor the caller-owned matrix example promotes native anyon
execution or closes M3.

Run the public source construction example:

```bash
python -m examples.ising_braid_source
```

Expected output:

```text
Fusion paths: (('1', 'sigma', '1'), ('psi', 'sigma', '1'))
Independent snapshots: True
Charge observable: fusion_psi
Projector: (2, 2) torch.complex128 cpu
```

Final supported-Python compatibility selections pass 208 cases on each of
Python 3.10, 3.11 and 3.12, including 62 real-source consumers, the unchanged
baseline and existing Rydberg/native paths. A separate source/baseline/public
example selection passes 126 cases on each interpreter. Strict mypy, using the
existing CI command with an explicit Python 3.12 target, passes 829 package
files and the new test file. Floor/current Black and Ruff and twelve repository
checks pass without exclusions or new suppressions.

The first default invocation had 15,053 passed, 169 skipped and one failure:
the new module's executable docstring had not been enrolled in the existing
doctest list. Enrollment was added without removing the example or weakening
the gate. A separate producer/reader probe found that an extended context
subclass could pass verification yet emit an unreviewed field that the reader
refused. Its added regression failed before the fix; exact frozen-profile
admission now rejects the extension before encoding or frontend return.

Final restricted-environment runs also retained negative evidence: the default
tier reported 15,044 passed, 168 skipped and 12 failures; Runtime integration
reported 1,844 passed, 125 skipped, three failures and 13 errors. These were local
socket preflight/Gloo cases blocked by PermissionError. Complete tiers are being
verified with local sockets allowed, with the same code and assertions; the
restricted invocations are not recorded as passes. The final complete default
tier with local sockets allowed passed 15,056 cases with 168 original skips in
284.07 seconds; its XML has zero failures/errors across 15,224 cases. The final
complete Runtime integration tier passed 1,860 cases with 125 original skips in
165.42 seconds, also with zero XML failures/errors. Both use unchanged tests and
assertions. Remote latest-head CI/GPU remain separate gates, and this PR has no
inherited merge authorization.

## Simulation finite-action delivery

The Simulation delivery is stacked on the Core source PR #767. It adds the
internal `simulation/ising_braid.py` action; no public export, IR field, plan
schema or result type changes. Core's existing verifier admits the exact native
profile before numerical work. F/R matrices use the reviewed complex gauge;
each clockwise generator is its adjoint. Ordered actions reuse SuperOperator's
left/right multiplication, with no dense superoperator allocation, matrix
exponential, Hamiltonian logarithm or numerical-time interpretation.

The independent tests use literal complex R entries and the closed form of B1,
rather than repeating the implementation's F-dagger R F construction. They check
both adjacent braid relations, distant commutation, all three generator
inverses, both initial charges, complex coherences, and a 100-exchange word at
atol=1e-12, rtol=0. Four outer exchanges yield the unitary -i I while leaving a
coherent density unchanged; the source words remain distinct. Fresh matrices,
independent results, CPU placement under a foreign default device and unchanged
random-generator state are also checked. Root execution remains explicitly
refused pending Runtime admission, and the baseline consumer factories remain
unchanged.

The following runnable public mathematical probe illustrates the same finite
action and the expected charge probability. It is a caller-owned matrix
reference, not an activated native execution frontend:

```python
import math
import torch
from flagquantum.experimental.simulation import IsingBraidProgram
from flagquantum.operators import SuperOperator

program = IsingBraidProgram().exchange(1)
fusion = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
exchange = torch.diag(torch.exp(torch.tensor(
    [-1j * math.pi / 8, 3j * math.pi / 8], dtype=torch.complex128
)))
middle = fusion.mH @ exchange @ fusion
state = SuperOperator.left_right_multiply(middle, middle.mH).apply(
    program.projector("1")
)
probability = torch.trace(program.projector("psi") @ state).real.item()
print(tuple(state.shape), state.dtype)
print(round(probability, 12))
```

Expected output:

```text
(2, 2) torch.complex128
0.5
```

The initial 50-test run had one invalid other-profile fixture: its Rydberg
minimum separation was zero, so Core correctly refused it before the intended
Ising admission assertion. The fixture now uses a valid positive separation;
no production validation, scientific tolerance or assertion was weakened. A
typing failure in deliberate malformed-context construction was corrected by
using an explicitly arbitrary-argument test callable. Final local verification
passes 51 new action cases and a combined 136-case source/kernel/baseline
selection on each of Python 3.10, 3.11 and 3.12. Strict CI-targeted mypy passes
831 files (830 package files plus the new test), floor/current Black and Ruff
pass, and all twelve repository checks pass. The complete default tier passes
15,107 cases with 168 original skips in 279.43 seconds; Runtime integration
passes 1,860 cases with 125 original skips in 147.38 seconds. Both XML reports
have zero failures/errors, with local Gloo sockets allowed and no test changes.
The public mathematical probe above was executed with the quoted output.

Remote latest-head applicable checks and complete CI/GPU remain separate gates.
The owner separately authorized #767's normal merge only after all such checks
and repository rules pass; that grant does not cover this Simulation PR.
