# Target-position dispatch source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `apply_gate_to_lth_target.ipynb` | `7b733722e6587cd31d805804d46e66f18f83a53a` | Q-0582 |
| `apply_gate_to_lth_target.py` | `77f70157ced3fcd8195951f16a34d16a8cfd709a` | Q-0583 |
| `apply_gate_to_lth_target_test.py` | `79e770fc3b13e4cfd0e8c4ffa21a14a8f100bee6` | Q-0584 |

The [native tutorial](../../examples/notebooks/apply_gate_to_lth_target.ipynb)
covers the complete original 13-cell sequence. It shows the original
two controls, three-bit selection with bound four, four target positions,
signature and wire labels, actual primitive compilation, and the six direct
call identities. The circuit runs through `circuit.append(op)`.

The compiler's target-position binding composes the shared unary traversal
with the selected physical target. Selection axes are big-endian
individually and their combined index is row-major. Each child receives
the flattened index, a positive predicate, and the exact target wire.
The caller supplies initialized parent work and distinct classical slots.
Unselected targets retain their input, and the callback supplies the
selected child and its adjoint when reverse traversal is requested.
The Pauli entry uses that binding and elides identity emission. The source
call graph still counts both controlled identity leaves.

## Original cases and execution

The entire locked suite yielded 13 passes including the independent
notebook and one expected failure for callable example serialization.
The native suite covers the width/extent pairs (3, 5), (3, 7) and (4, 5).
It compares the original `make_on` inferred source wire order and
signature to the native operands, preserves the source diagram labels,
and walks the source primitive stream with measurement feedback for
the controlled X and odd-Z examples. The two-control tutorial fixture
checks a coherent superposition of all 256 valid basis rows in all 16
raw measurement branches. The reverse traversal restores the state.
A controlled H/X callback on a nonmonotonic target bank adds a
non-Pauli witness with exact source operands, both measurement
outcomes, and complete complex output including clean work.

| Tutorial fixture | Emitted forward primitives | Declared source model |
| --- | --- | --- |
| I/Z/I/Z, two positive controls | 16 T/T†, 27 CX, 6 CZ, 4 measurements, 4 resets | 4 And, 9 Clifford, 4 measurements, 16 T equivalents |
| Wire width | 13 reserved, including four clean work wires | 12 peak live wires |

The extra reserved wire comes from the selection register's three physical
bits and its four-address extent. The third address level contributes no
live predicate on this fixture. The decorated tutorial graph and independent
cost query are verified separately from emitted counts.

## Remaining source-file boundaries

Q-0582 and Q-0584 are accepted. Q-0583 retains arbitrary foreign gate
objects and foreign serialization as open compatibility boundaries.
Native callers provide controlled child recipes and explicit wire
bindings. The locked source itself rejects symbolic iteration extents
when resolving target registers or diagram labels. The package
initializer Q-0581 remains open while its other exports are audited.


## Complete callable source bindings

The complete `ApplyGateToLthQubit` object is captured through the existing
Ecosystem named-register emitter. Foreign gate and callable objects end at that
boundary. Restored native JSON snapshots replace the emitter behind the existing
`SelectBlockEncoding` consumer, with no new production abstraction. Users append
the bound native operation to `Circuit` in both original source directions.

The same index callable supplies alternating H/S and S/T leaves, with zero,
one or two positive control bits and named one- or two-axis row-major selection.
The expected logical operator is built independently from the selected target's
single-qubit matrix and identity bystanders. Every raw measurement history is
checked for a coherent full-register input. The two-target cases with zero or
one control additionally check every operator column. All work returns to zero,
including the original inverse's measured controlled-phase decomposition. Native
IR preserves every captured instruction, operand and feedback and round-trips
through JSON. Concrete widths are bound before capture. Unbound symbolic widths
and symbolic iteration extents keep their explicit admission/source failures.

The independently planned indexed model reuses the existing controlled-leaf
costs and source whole-operation QEC and peak-width comparisons. It never counts
snapshot instructions as model estimates.

| Target-position binding | Captured primitives | Independent model basis |
| --- | --- | --- |
| Forward H/S or S/T | 4 T-family per traversal erasure, controlled phases stay atomic | Source controlled leaf models plus bounded traversal |
| Original inverse | One extra erasure per S/T leaf, 4 T-family per erasure, plus one T-family per T leaf | Source child adjoints and independent traversal model |

For `N` selected targets and `c` positive controls, the fixtures emit `N+c-2`
forward measurements. Inverse H/S adds `N/2` and inverse S/T adds `N`.
The source model includes controlled-S/T And costs in either direction, so its
estimated And/measurement inventory is intentionally recorded separately from
the captured forward primitive count. The original source callable's own foreign
serialization remains the previously recorded expected failure. General foreign
gates beyond the supported importer vocabulary remain open under Q-0583.

## Complete callable and matrix responsibility, November 28

Whole-file review covers all fields/converters/defaults, make_on inference,
control/selection/target properties, diagram/index mapping, nth_operation,
example and documentation record. The source callable is evaluated only at
conversion; native snapshots retain neither the callable nor SDK objects.
Selection is MSB first and multiple axes flatten row-major. Logical words
remain caller-owned; captured unary traversal owns initially clean work and
measurement slots, returned clean.

The shared importer already supports a callable returning a general one-qubit
MatrixGate together with Rx. test_target_position_matrix_consumer.py now
verifies that actual source path with zero, one and two controls, both source
directions, every complete operator column and every raw measurement history.
The matrix payload survives JSON restoration and the restored instructions
execute through ordinary append with zero private work at return. This reuses
the accepted Majorana generic-gate dependency; no production emitter is copied.
The first probe attempted dataclass equality on tensor payloads, which raises
ambiguous Boolean conversion. The final check verifies exact JSON roundtrip
and executes restored payloads instead; that failed probe is not evidence.

For zero/one/two controls the actual snapshot has zero/one/two measurements;
the latter two emit four/eight T-family gates and seven/thirteen CX. All retain
one controlled Rx and one two-wire matrix leaf. Independently, source models
report two/three/four And, five rotations and two/three/four measurements.
Those matrix/rotation categories are not synthesized Clifford+T instructions.
Directional records remain in g2-indexed-matrix-resources-20261128.xml.

Under the approved [G2 closure rule](G2_FAITHFUL_MIGRATION_CLOSURE_PROPOSAL.md),
Q-0583 can close after batch acceptance. Callable foreign serialization, source
symbolic-extent refusal and unbound-width refusal remain explicit dependency
limits. The former restricted-gate note above is superseded for the accepted
one-qubit unitary importer domain. No arbitrary channel or invented source
primitive decomposition is claimed. Scalar phases without a source quantum
operand remain distinct from native carrier bookkeeping.

Final batch verification: 611 concentrated cases and 611 independent Cirq 1.6
cases pass. Three additional complete matrix-callable operator cases pass in
each lane, covering both directions and every applicable raw history. The two
unchanged original files report 21 passes and one original expected failure;
one notebook is deselected and reuses its earlier complete receipt. The
unchanged integration owners reuse 248 passing current-branch cases. Three
Python owners/tests pass strict typing, Black and Ruff. Architecture, IR, API,
required/Cirq contracts, Algorithms scope, English, links and whitespace pass.
Six actual optional/mandatory missing-reference cases pass; mandatory errors
retain their original exception. The existing 568 dependency findings are
unchanged, with none on the new consumer. Source fixtures verify frozen files,
Qualtran 0.7.0 and the actual Kickmix 0.1 Simulator. Receipt prefix:
g2-indexed-source-closure. No production execution owner was changed.
