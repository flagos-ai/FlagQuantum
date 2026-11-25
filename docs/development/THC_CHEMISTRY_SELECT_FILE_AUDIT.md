# Tensor hypercontraction SELECT audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, under
`qualtran/bloqs/chemistry/thc`.

| Inventory | Source file | Git blob |
| --- | --- | --- |
| Q-0419 | `select_bloq.py` | `a3b825950fc09a429f829637ed14d7c73e922bb6` |
| Q-0420 | `select_bloq_test.py` | `f1b2b7a84dc7d831216470214ca02f6b07e02349` |

SELECT declares bounded orbital addresses and six selection flags
alongside the alias precision word and the two opaque spin banks.
An optional external control bit precedes the selection ports.
Each of the four directional rotation leaves has a one-body flag,
an angle-data word, an address word, and one spin bank. The tests
match the ordered ports, dtypes, domains, and register directions.

The source allocates one angle word, passes it through all four
rotation leaves and frees it. Tests identify the allocation and
release and compare the intermediate angle connections. The model
now derives both SELECT and leaf widths from their source
signatures, keeping the shared angle word live at the top level.
For either external control polarity, the original composite
returns the bit directly from the input. The model includes the
extra live bit and retains the original gate categories. Its tests
cover both polarities and the uncontrolled case at four geometries.

## Emitted operations and source limits

The executable SELECT continues to use `circuit.append(op)`.
The exact and eight-bit paths take the existing Givens networks,
compare their complex amplitudes, and clear the reused work. The
unblocked and blocked lookups exercise both coherent and measured
cleanup. In the measured fixtures the default arithmetic emits
32 or 64 measurements respectively. The upstream-style carry
path adds 28 measurements and changes the emitted Toffoli count.
The walk tests check feedback-slot ownership and complex phases
in both directions.

The reference rotation leaf gives scalar Toffoli counts and
does not supply a decomposition. The representative source model
with rank two, four spin orbitals and eight angle bits estimates
107 Toffoli and 11 controlled swaps, with 24 peak logical qubits.
This is separate from the physical instruction and measurement
counts. The optional external control on the source declaration
does not implement an independently controlled executable SELECT.

The locked original tests returned three passes and two skips.
The frozen 40-cell tutorial was already executed independently
during PREPARE acceptance. Q-0420 closes the original tests.
Q-0419 remains open for the source rotation decomposition and the
independently controlled native binding.

The concentrated native regression passed 185 cases. Strict typing,
format, ownership, architecture, IR, and public API checks passed.

## Complete source constructor and typed dependency binding, 2026-11-26

All definitions in the locked `select_bloq.py` were reread. The actual example
overwrites `num_mu=8` with 10, then uses eight spin orbitals, twelve angle bits
and ten sampling bits. The constructor adapter preserves this example and the
original DOC binding. Original control dispatch selects the corresponding
`control_val` declaration; neither polarity controls the source body.

Shared port ownership is now `registers/layout/thc_signatures.py`. Formats and
the existing Algorithms signature helpers use it. Algorithms retains its
existing positive, paired-spin, minimum-angle and binary-control validation,
including the original exception behavior. The source adapter separately admits
original odd-spin/one-angle-bit logical declarations; it does not promote them
to that numerical API or assign them a repaired resource model. Empty source
swap words retain declarations and the original `CSwap does not accept Soquets`
decomposition counterexample. Symbolic angle/sampling words and symbolic system
widths retain their successful logical graphs. The original depth-two symbolic
system export fails at Split and remains a documented refusal.

`registers/control/thc_selection_component.py` owns the exact fifteen source
calls and their named operand/return wiring. One owned angle word is allocated,
passed through all four directional rotation declarations, then clean-freed.
The first pair uses the full table; the second pair sets `two_body_only`.
Both block factors and direction flags retain the original constructor values.
System swaps, positive flag/address swaps and both first-system-bit phase
predicates keep their original order and polarity. External control, sampling
and rotation flags that the original does not use return the same input SSA
values. No angle table is invented for the rotation placeholder.

`registers/primitives/controlled_z_word_component.py` owns the direct generic
phase dependency, its signature and source Split/MultiControlZ/Join wrapper.
It phases the MSB of one through-system word and preserves other system bits
and shaped controls. Predicate work and erasure remain with MultiControlZ.
The existing `ChemistryNumberPhase` delegates its signature here without changing
its append implementation or model. Formats supplies explicit factories for
SelectTHC, THCRotations and ApplyControlledZs. Their metadata remains in the
original input ComponentLibrary, separate from executable Program definitions.

Actual source wire depths zero, one and two, nondefault block factors, both
control polarities and the original factory compare every recursively comparable
call and operand. The dedicated constructor preserves the literal call order.
The existing generic stored-graph width scheduler may reorder independent calls;
its unmodified schedule is not claimed to be the same ordered stream. Three
frozen source/test/dependency blobs, Qualtran 0.7.0 and real Kickmix 0.1 Simulator
are verified before the source cases. Missing rotation bodies reject physical
binding before input/work allocation; they are never treated as identities.

The direct phase wrapper compares its actual primitive gates, operands,
measurements and feedback against the unmodified original decomposition. All
small raw histories retain complete complex amplitudes and zero private work.
Normal `circuit.append(op)` runs in trajectory and batched mode on arbitrary
coherent logical states, including system spectators. Earlier complete native
eight-bit THC arithmetic, network, SELECT and walk evidence is reused separately
from the unresolved literal source rotations.

| Two-bit system | Actual primitive view | Original wrapper quote | Original/actual peak |
| --- | --- | --- | --- |
| One control | One CZ, no private work or measurement | Zero quoted Toffoli; no quoted Cliffords | 3 / 3 |
| Two mixed controls | 4 T-family, 6 CX, 2 CZ, one measurement/reset, one clean work bit | One Toffoli | 5 / 5 |
| Three mixed controls | 8 T-family, 12 CX, 3 CZ, two measurements/resets, two clean work bits | Two Toffoli | 7 / 7 |

The wrapper deliberately quotes only Toffoli costs. Those estimates are not
literal Clifford/feedback inventories. For rank two, four spin orbitals and
eight angle bits the source SELECT model remains 107 Toffoli, 11 controlled
swaps and one Clifford, with peak 24 or 25 including an optional outer control.
The literal root has no actual primitive inventory while rotations are absent.
A published-baseline comparison proves all 84 existing THC signatures/cost
trees and six phase signatures/models/appendable operations remain identical.

Q-0419 remains open for its existing rotation-body and independently controlled
native requirements. Source-only models are retained under their original
scope; completing constructor resolution does not certify those physical gaps.
No Core operation, public signature, serialized vocabulary, backend choice or
pending proposal is changed by this batch.


Batch acceptance: 208 concentrated cases pass in each Cirq lane, plus the
additional SDK-blocked constructor case passes separately in both lanes.
Local integration passes 145 cases. The unchanged original complete file tests
pass three and skip two source-owned checks (decomposition-derived counts and
strict BQUInt-to-QAny type compatibility). Strict seven Python files, Black,
Ruff, both owner scopes, language/links, architecture, IR and public API pass.
The three newly bound identifiers belong to the frozen 303-entry resolver,
whose blob is independently verified. This batch does not infer the global
complete-constructor total from registration presence: the old 117/303 display
was a historical snapshot preceding later sparse/Hubbard receipts, and complete
aggregate constructor reconciliation remains an explicit Q-0876 obligation.
