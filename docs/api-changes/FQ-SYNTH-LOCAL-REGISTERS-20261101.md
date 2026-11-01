# Local register ownership, cleanup and physical reuse

Date: 2026-11-01. Continue sixth-plan lifecycle work after #991, merged at
e0993b6a9a63d5eedd425cf70e956dc353675684 with 57 successful checks and three
configured skips. Only private Compiler construction behavior is extended.

## Ownership contract

qreg.local produces one fresh quantum.register reference with a nonempty unique
name, no operands and no region. It owns temporary storage rather than a named
output. Existing compute/phase/Const calls and exact uncompute apply unchanged.
qreg.release(effect, local) -> next_effect has no attributes/regions; it requires
a live owned local reference in the fresh proven-zero state. Release invalidates
the reference and returns its zero physical storage to the local pool. It never
accepts initialized storage, an output, a Const borrow, a user zero assertion,
extra arguments, or a previously released reference.

Every local must be released before return and cannot escape in the output list.
Outputs and borrows retain their existing initialization/order obligations.
Initialization of a local is not proof of subsequent zero; clean direct computed
locals with the #991 selected inverse first. A never-initialized fresh local can
be released. No arbitrary helper Output inverse is inferred. Locals can be Const
actuals when initialized; released or fresh reads fail. All definitions verify,
including unused helpers, and the expanded root verifies again.

## Storage, binding and accounting

Named outputs keep their contiguous MSB-first mapping. Locals occupy a separate
internal band after all outputs. Assign the lowest available zero local slots
deterministically, allowing released slots to serve later local references;
different simultaneously live references remain disjoint. No new register type,
allocator record or public report field is introduced.

Arithmetic scratch starts after the entire high-water local band for the whole
program. Released local slots do not become arithmetic scratch. This conservative
separation is required: a saved forward candidate's exact inverse must find its
original scratch clean even if another local is allocated in between. Whole
workspace equals the reserved local-band peak plus the maximum segment scratch;
whole allocated width equals named-output width plus that workspace. Include
helper locals after expansion. Never omit hidden locals from qubit/workspace,
memory or target budgets, or mistake used width for allocated capacity.

Expansion clones each helper-local SSA identity and internal label per call,
including repeated/nested calls, preserving roles, effects and source locations.
Source/final-node and expansion-visit limits remain. Width validation includes
reserved future outputs and the high-water local band, not just the currently
encountered declaration prefix. No initialized/unknown storage enters a zero pool.

## Acceptance and boundary

Execute all sixteen two-bit inputs, complex superpositions, external reference
entanglement, absolute/relative phases and interference under both schedules.
Compare sequential/reused and overlapping locals' actual widths and constraints.
Check no named local escapes, zero internal work, repeated helper-local identity
isolation, final mapped resources, and replacement by an existing directly
borrowed phase producer behind the same Program/run/reader consumer. Execute raw
source instructions with nontrivial middle phases as well as checked targets.
In particular retain a computed result, allocate another local, then invert the
first result: scratch must remain safe. Test release initialization/role/liveness
refusals, malformed operations, stale effects, hidden-resource budgets and late
output declarations. Existing Python/native source entry and target guards stay.

This establishes private whole-register local ownership and verified zero-slot
reuse, not native local/within syntax, consuming Input, arbitrary state drop,
generic free assertions, arrays/views, general unitary control or complete
Classiq within compatibility. The four original Qmod files/licenses/digests stay
unchanged; independent trials remain 0/2 and release_authorized=false.
