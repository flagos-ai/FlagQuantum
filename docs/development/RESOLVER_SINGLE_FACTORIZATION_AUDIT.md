# Single-factorization stored constructor closure

Reference: complete unchanged Qualtran v0.7.0 `chemistry/sf` source files at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Classification:
`single_device_fast_path`.

## Interface and ownership recorded before implementation

Ecosystem translates the five original constructor identifiers and reuses the
existing Algorithm-owned signatures. Compiler owns literal one-body and block
composition, H/CSwap and the existing prepared reflection implementation.
All words are caller-owned, MSB first; these roots allocate no extra words of
their own. Child reflection work retains its own lifetime owner. Missing inner,
outer and SELECT physical bodies remain declarations, never identity gates.

One-body preserves nine calls and its unused is_adjoint field, distinct
forward/inverse QROAM blocking factors and l_ne_zero/rot_aa/sys bystanders.
Block preserves five calls, the original ordered one-body return unpacking,
the inverse outer call on rot_aa[1] instead of rot_aa[0], and the reflection's
single succ_l control. The separately implemented coefficient-driven paper
algorithm has corrected semantics and must not replace this original path.
Original alpha/epsilon remain unavailable. Finite constructor coverage does
not close Q-0405's whole physical source boundary.


## Verified constructor and dependency scope

All five exact identifiers resolve original stored level-0/1/2 libraries using
`ecosystem/_single_factorization_library.py`. The two roots use
`compiler/registers/preparation/single_factorization_component.py`; original
inner/outer/SELECT stay declaration-only. Codec round trips preserve root
fields, including distinct forward/inverse blocking factors and both one-body
is_adjoint values. SDK-blocked resolution, signatures, ordered source call
outputs, every named input producer and all final boundary outputs agree.
Missing physical bodies refuse binding. SELECT additional_control retains the
original unchanged five-port signature, with no system or extra control port.

Original zero-address inner/SELECT declarations serialize and resolve. Original
outer auxiliary-zero shallow metadata resolves, but levels 1 and 2 raise the
original KeyError for target in uniform preparation; this is an asserted
counterexample. Original small block fixtures can also fail QROAM annotation
with AssertionError. Valid original examples, not patched references, provide
the successful root comparisons. Zero-spatial-width root bodies remain absent.

## Validation

Both actual Cirq 1.7.0/Qiskit 2.5.0 and Cirq 1.6.1/Qiskit 2.0.0 profiles pass
171 concentrated existing/new cases, followed by fourteen final focused cases
including the added selected-reflection check (172 distinct cases). Actual
Qualtran 0.7.0 and Kickmix 0.1 Simulator are asserted by the launcher.
Local integration passes 145 cases. Three changed Python files pass strict
mypy, Black/Ruff and ownership checks; language, links, architecture, internal
IR and public API baselines pass. Six frozen source/test blobs are verified.
The unchanged original tests pass seventeen cases with twelve source skips and
one notebook deselected; prior notebook evidence is reused, not rerun.

The actual reflection selected by the new block root is extracted with the
existing reachable-definition utility. Its recursive comparable calls match
unchanged upstream. Accepted atomic Identity implementations remain terminals.
Its full 32-column complex operator matches the original tensor contraction,
every small raw erasure history agrees, all private output amplitudes vanish
and all workspace leases return. No unused opaque root definitions enter this
isolated physical check. The separately accepted coefficient-driven SF
executables are included in regression without claiming they are the original
opaque declarations.

## Actual primitives and source models

For the small selected reflection, the emitted inventory is 77 instructions:
12 T-family gates, three measurements and three resets, with three private
qubits at peak. These actual counts are separate from whole-root models.

For auxiliary=50, spin orbitals=10, alias bits=12, forward blocks=(4,8), inverse
blocks=(2,4), the original/native one-body model with rotation bits=7 gives
846 Toffoli, 18 CSwap, 46 And, 514 Clifford, 46 measurements and peak 75.
The block with outer/inner rotation bits=(7,5) gives 1737 Toffoli, 50 CSwap,
179 And, 1801 Clifford, four rotations, 179 measurements and peak 80.
Both complete seven-category and peak estimates match original owners.
`g2-sf-resources-20261127.json` retains the raw separate inventories. One-body
has three opaque definitions and block five; whole primitive inventories are
unavailable, not zero. Q-0405 remains open. No original bug is silently fixed
or waived and no alpha/epsilon value is invented.

## Complete factorization source closure (2026-11-28)

The [shared factorization closure](RESOLVER_DOUBLE_FACTORIZATION_AUDIT.md#complete-factorization-source-closure-2026-11-28)
adds the original SF block GenericAdjoint, symbolic precision declarations and
source-success symbolic one-body/block estimates. Imported child identities keep
outer preparation directions distinct. The original specialized one-body
adjoint retains the same nine-call body and its asymmetric blocking metadata;
the existing return permutation, second outer rot_aa connection and omitted
reflection control remain literal. Unknown inverse constructors still refuse.

The new scenario test is `tests/unit/test_factorization_source_closure.py`.
Original numerical optimizer failures and missing physical SELECT/prepare leaves
remain separate from source logical/model acceptance. Existing full-complex
selected reflection, every small raw measurement history, private cleanup and
actual 77-instruction/12-T-family/three-measurement inventory are reused.
No whole-root physical execution or alpha/epsilon value is invented.
