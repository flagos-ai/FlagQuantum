# Complete physical builder binding

## Intake and ownership contract

The complete circuit_builder.pybind.cc, circuit_builder.pybind.h and original
circuit_builder_test.py belong to K-0180, K-0181 and K-0182. All 62 recursively
quoted source blobs match frozen revision
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`. All six original tests pass before
native changes. The frozen reference remains unchanged.

Compiler formats owns mutable physical records, named register membership,
freezing and reusable INFO_TAG contexts. Compiler register layout owns growing
physical ID allocation. RegisterWorkspace remains authoritative for live leases
and reusable wires. Existing gate, arithmetic, lookup, finite-field and unary
binders own their decompositions. Numerical execution remains in Simulation,
with sessions in Ecosystem and rendering in Drawer.

Before implementation, the allocation contract is as follows. Quantum and
classical IDs have independent increasing cursors. Temporary allocations reuse
the last individually freed wire first. Array free converts the complete input
before processing its members sequentially, so a later invalid member can leave
an earlier valid release committed. Named registers reserve new IDs without
entering the temporary free pool. Quantum named members contribute to the
quantum budget. Source classical named members do not enter its temporary
allocation counter. Source named members cannot be freed through temporary
ownership. These source asymmetries are retained explicitly.

Physical operand order follows allocation order. Named register members and
returned immutable OperandArray views share values, with independent lifetime
from a later freeze or builder clear. Allocation emits no reset. The caller
must restore clean wires before release. Allocation and metadata add zero
quantum primitives and zero measurements. Static instruction counts and actual
executed counters are recorded separately. This boundary has no analytical
resource model.

Mark contexts own their builder and immutable Python label. They are reusable,
append an enter and exit on every use, propagate body exceptions, and preserve
literal out-of-order exits for the existing renderer's best-effort handling.
Python ownership replaces the foreign move-only string-view storage contract.

Native format limits and bounded allocation remain enforced before large
materialization. Source size_t overflow and malformed foreign storage are
outside admitted native values. The complete source parent remains open until
all mapped dependencies and consumer observations are verified.

## Complete declaration map

| Original definitions | Native ownership and disposition |
| --- | --- |
| Constructor and finish_circuit | KmxCircuitBuilder and freeze retain independent frozen values |
| max_qubits, num_free_qubits, num_allocated_qubits and set_max_qubits | Existing builder owns the quantum budget. SIZE_MAX is unbounded. Public free count retains the signed-int64 projection, including large-limit wrap. Positive float32-overflow values select infinity as in the source |
| alloc_qubit, alloc_bit, alloc_qubits and alloc_bits | WireAllocation shares RegisterWorkspace leases and accepts the original integer conversion protocols. Quantum and classical cursors remain independent |
| free_qubit, free_bit and free | Existing workspace releases each converted member sequentially. Named IDs have permanent ownership and cannot enter temporary free stacks |
| create_quantum_register and create_classical_register | Existing builder columns own ordered names and members. Allocator reserves new IDs independently of reusable temporary wires |
| mark, mark enter and exit | Reusable builder-owned contexts append literal INFO_TAG records. Exceptions propagate and mismatched manual exits remain observable |
| x, z and cx | Existing broadcast_binding and mixed primitive emitters |
| ccx, init_and and del_and | Existing broadcast_ccx, source lane dispatch and measured-AND ownership |
| cz, ccz, swap and cswap | Existing broadcast_cz, broadcast_ccz, broadcast_swap and broadcast_cswap |
| z_pow | Ecosystem phase_angle_from_half_turns returns the existing exact PhaseAngle128. Compiler KmxRecord owns literal phase records. Compiler does not import the Ecosystem converter |
| neg, neg_if, push_condition and pop_condition | Existing exact physical records and lowering preserve guards and nesting |
| bit_store | Existing classical-store binding and physical record lowering |
| debug_print, reset and hmr | Existing physical debug codec, reset_x_register_ir and measure_x_clear_register_ir retain ordered lanes and feedback slots |
| init_lookup and del_lookup | Existing lookup_binding and accepted forward/erasure generators use explicit caller workspace. Complete ordinary tables pass. The original dynamic integer-subclass boundary and failed-wrapper temporary allocation remain open |
| iadd and isub | Existing physical_addition_binding retains bounded tolerance and control dispatch |
| left_rotate and right_rotate | Existing bound_bit_rotation_ir owns signed shift and endian adaptation |
| multi_controlled_x | Existing physical_multi_controlled_x owns controls, target and workspace |
| flip_if_less_than, flip_if_greater_than and flip_if_equal | Existing physical_comparison and bound_quantum_equal_ir own relation, tolerance, enable and X-output dispatch. Original empty-workspace failure remains registered |
| gf2_iadd, gf2_imul and gf2_idiv | Existing field_addition_binding and field_constant_binding retain field selection and literal scalar arithmetic |
| init_gf2_mul, ixor_gf2_mul, del_gf2_mul and gf2_phase_by_product | Existing field_product_binding and field_product_phase_binding retain reset, control and measured erasure |
| init_gf2_inverse, init_gf2_inverse_with_scaffold, del_gf2_inverse and del_gf2_inverse_with_scaffold | Existing field_inverse_binding retains explicit target and scaffold ownership |
| init_gf2_div, init_gf2_div_with_scaffold, ixor_gf2_div, del_gf2_div and del_gf2_div_with_scaffold | Existing field_division_binding retains mode and scaffold ownership |
| unary_iteration | Existing UnaryIterationPass retains context and iterator lifetimes through the accepted cursor owner |
| info_tag and owned mark name storage | Existing owned Python strings and raw physical metadata columns |
| Foreign registration, no-copy policy and repr | Native module ownership, independent builder copies and native representation replace extension registration and foreign string-view lifetimes |

The header's private classical counter has no remaining native consumer.
Classical temporary live ownership is recorded by the workspace itself.
Permanent classical IDs remain outside the temporary free pool. This retires
the unused counter field while preserving every observed source behavior.
Native allocation-limit diagnostics use native vocabulary. Huge allocations
outside native format limits refuse before materialization. Source allocation
failure mutation, invalid pointers and source SIMD performance are separate
boundaries. Full K-0180 stays open for these registered dependency gaps.

## Executed consumer evidence

The source generator executes 63 lifetime transactions through the unchanged
extension, recording each returned operand, quantum budget, live quantum count,
free count and full frozen circuit after success or failure. All observations
execute three lanes and retain canonical KMX, complete ordered registers,
reaction depth, exact phase, quantum/classical output and separate static and
actual counters. It also records reusable, nested, empty, failed and manually
misordered marks, exact decimal phase and debug output consumers. Fixture
regeneration must remain byte-exact.

The original six tests run unchanged in the locked extension. Native tests
retain each original scenario through the mapped owners, including exact
float/string phase differences and XML-escaped flame labels. Copy/clear
independence, immutable returned views, integer conversion protocols, budget
float32 boundaries, format limits and conversion-before-release are separate
native checks. The existing full complex and all-small reset/HMR branch
consumers are replaced by incremental construction and mark extraction while
keeping their assertions intact. The normal circuit.append(op) example uses
the same replacement.

This batch fixes an actual dependency failure. Raw INFO_TAG and debug records
were passed to reaction_depth_for_operations, which attempted to read a missing
quantum target. The original source ignores these records. The existing shared
analysis owner now follows that rule. A direct unchanged-source comparison and
the complete marked builder observation cover the regression.

The batch passes 26 focused scenarios, 4,521 concentrated regression items and
121 local construction/coherent/trajectory integration items. Strict typing
passes all seven touched Python files. The source fixture regenerates
byte-exactly and all 62 source blobs are reverified after execution. Black,
Ruff, Compiler scope, architecture, multi-level IR and stable API checks pass.
Dependency policy retains the identical 720-entry baseline.

An additional 282-item coverage run measures 100 percent statement coverage of
WireAllocation, 96 percent of the complete mutable builder, and 98 percent of
reaction-depth analysis. The combined measured scope is 95 percent. Untouched
GenerationWorkspace methods in the shared workspace file are outside this
batch's coverage selection. This is statement coverage, with branch behavior
covered by explicit lifecycle scenarios rather than a whole-repository coverage
claim.

K-0181 and K-0182 are accepted as complete header ownership and original test
consumer disposition. K-0180 remains open for its registered direct dependency
and automatic-wrapper mutation gaps. Pending Proposals 116 through 119,
Q-0910/Q-0911 numerical cycle rounding and previously recorded source defects
remain independently open. Reference receipts use g2-builder-binding-*20261123
beside the worktree.
