# Simulation private-binding retirement

Implemented on CI-qualified PR1335 head
`f986fb4026b46a585a165becaaae52f8f3bce6cc`: 243 successful checks and three
existing workflow-conditional manual-pool skips. The exact parent Main and
Local GPU Gate workflows both succeeded; this child requires its own CI. This completes the identified unused private
aliases within Simulation, including the
[Dynamics normalizer retirement](dynamics-normalizer-retirement.md).

Fifteen Dynamics normalizer aliases and fourteen Statevector/Stabilizer aliases
are removed without forwarding. Existing definitions remain in their current
owners. The ledger retires imported aliases with empty definition lists so
original frozen definition-source identities remain unchanged. Its cumulative
545 rows and existing 102 public-entry inventory introduce no compatibility.
Six unused undeclared Dynamics imports are removed separately. All operational
consumers, including benchmark producers, import the defining modules directly;
raw benchmark records and scientific evidence are not rewritten.

The seven declared Lindblad exports, public circuit/engine APIs and historical
serialization identities remain. Existing required public payload locators,
including `observables._PauliTerm`, retain their itemized compatibility basis.
No numerical definition, precision, RNG order, public signature or schema is
changed. A broader unused-private-import inventory is a review list, not a claim
that every candidate can be removed: protected payload globals and directly
PR1084-dependent Compiler/Ecosystem work keep their existing gates.

AST invariance, all affected numerical/dispatch/conversion tests, historical
payloads, installed native artifacts, frozen scientific evidence checks and
formal full regressions must qualify this expanded batch independently of the
previous narrow preview. Its own pushed-head CI remains required.

See [the formal acceptance record](simulation-private-binding-acceptance.md)
for actual-checkout regressions, installed artifacts and incoming obligations.
