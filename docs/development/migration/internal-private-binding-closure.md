# Independent private-binding closure

Implemented after PR1336 parent
`4f04ec407066f7b93582713d05a867b6cc4f6bee` qualified: 243 successful
checks and 3 existing workflow-conditional manual-pool skips.
This child requires its own exact-head CI qualification. This batch combines nine QEC alias retirements
with fourteen unused Compiler Topology aliases and two internal Runtime Dynamic
aliases. Existing numerical/compiler definitions stay in their existing owners;
no private forwarding is added. Three test consumers import existing topology
helper owners directly. The two Runtime names are removed only from the
internal package namespace and its internal export list. Protected
`flagquantum.dynamic` and experimental task entry points are unchanged.

The cumulative ledger has 560 rows. Retirement-only rows have empty definition
lists, preserving all existing source-definition identity obligations. The
existing 102-entry historical facade inventory remains unchanged; it is not an
exhaustive inventory of the separate QEC frozen globals. Thirty-three QEC
pickle, namespace and replacement-owner obligations are itemized in the
[QEC subset record](qec-private-binding-retirement.md). They are retained rather
than weakening existing contracts, historical fixtures or assertions.

The original 98-binding unused-private-import review is accounted for as:
29 Simulation aliases retired by the parent, these 25 proposed retirements,
33 existing frozen QEC globals and the required Observable `_PauliTerm` payload
locator retained, and ten directly PR1084-dependent frontend/Ecosystem aliases
still gated with their complete Construction/IR family. This inventory is not
an exhaustive audit of the whole architecture migration.

The original narrow QEC source/wheel/install preview remains preserved. The
expanded source, full original tests, public and historical payload behavior,
incoming-consumer audit and native installed artifact must qualify independently.
No public API contract, scientific threshold, RNG order, raw result, native ABI
or optional dependency policy changes. Construction/IR, directly dependent
compiler work, native/development distribution and Rust retain their gates.

## Obsolete internal alias test parameters

Three existing routing tests parameterize both actual/public dependencies and
otherwise unused private aliases. Ten cases require the exact private aliases
this authorized cleanup retires: three emission aliases, six strategy/trial
aliases and the unused restoration-path alias. Only those parameter values are removed. All three test bodies and their
identity assertions remain unchanged, with all actual/public dependency cases
retained. Numerical/statevector assertions and public routing evidence/pickle
checks remain unchanged. These are internal compatibility cases, not frozen
public/scientific fixtures; the public API protection policy explicitly excludes
private helpers absent a separate contract. The three initial emission failures,
six initial strategy failures and the full-suite restoration alias failure are retained as the original diagnostic.
The retired root `_routing_metadata` inventory now names its actual emission
owner rather than a second alias that no longer exists. No contract is edited.

## Internal consumers of retained compatibility globals

The retained QEC namespace and pickle globals remain published entry points,
not internal dependency shortcuts. Belief-propagation validation imports
`_count` and `_normalized_indices` from `dem/model.py`; the repetition workflow
imports `_repetition_wire` from `decoders/repetition.py`. All three are the same
existing function objects, with the same registered pickle identities.
No helper definition, parameter, public export or numerical behavior changes.
The no-reference-allocation scenario patches `execution/signatures.py`'s actual
`_executed_flip_set` owner rather than the unused construction alias, preserving
all assertions while guarding the real callable. Frozen metadata/namespace
compatibility tests continue to exercise the published surviving entry points.

See [the formal acceptance record](internal-private-binding-acceptance.md) for
actual-checkout regressions, native installed evidence and integration gates.
