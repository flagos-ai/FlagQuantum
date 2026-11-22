# Complete physical operation statistics audit

## Ownership and interface before implementation

The frozen operation_statistics.pybind.cc/.h are K-0246/K-0247 at
9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453. The complete implementation,
all 19 groups, mapping operations, equality, documentation examples and both
snapshot factories are in scope. Direct circuit and interactive counting
consumers retain their own complete-file acceptance decisions.

Compiler owns the immutable primitive-count projection in formats and static
KmxCircuit inspection. The existing PhaseAngle128 owns exact angle classification
and OPERATION_CODES owns raw conditional opcode identity. Two immutable tuples
retain all 256 raw counts and 129 rotation levels. Existing KmxCircuitStats
cannot replace them because its grouped histogram erases conditional identity
and does not retain phase levels. The projection is internal to the existing Compiler physical-format boundary.

Simulation owns lane accumulation and a narrow optional execution observer on
apply_packed_basis_records/sample_basis_batch. It receives the existing owned
record protocol and an active lane bit mask before quantum controls modify it.
Lane zero is the least significant mask bit. Metadata condition-stack and tag
records count every lane. Each counter wraps at 64 bits. Snapshot returns only
owned primitive tuples. Reading a snapshot transfers no state or work ownership.
The caller owns the packed state and counters independently, so resetting a shot
preserves counters and explicit counter clearing leaves the state untouched.

Compiler receives primitive counter tuples through its existing physical-format
boundary. Simulation consumes the existing record protocol. Original objects
remain inside reference tests. User circuit
execution continues through circuit.append(op). Analytical estimates remain
separate from static primitive counts and actual executed counts.

## Frozen reference evidence

The intake receipt verifies 32 unchanged source/header/test blobs. Three original
resource, mapping and simulator-counting tests pass. The native extension executes
all 129 exact phase hierarchy levels, 90 signed-int64 range queries and batch sizes
1, 3, 4, 64, 255, 256, 257 and 300. Source equality distinguishes conditional
opcodes and different phase levels even when grouped mappings are identical.
The reference files remain unchanged.

## Complete definition disposition

| Original definitions | Native owner |
| --- | --- |
| OP_COUNT_GROUPS, group_count | Compiler format group projection, existing OPERATION_CODES |
| from_circuit | PhysicalOperationStatistics.from_circuit, exact PhaseAngle128 classifier |
| from_simulator, enabled/index validation | Simulation counter snapshot, native primitive pair conversion |
| num_toffoli_like, num_non_clifford, num_t_like, num_instructions | Immutable snapshot properties, unsigned 64-bit sum semantics |
| num_z_pow, z_pow_clifford_hierarchy_counts | Inclusive clipped range and independent populated-level dictionary |
| get_item, get_with_default, contains, len | Native read-only mapping, zero/default identity and unknown-key distinctions |
| keys, values, items, iterator | Source group order, fresh owned lists |
| equality and inequality | Complete raw opcode and phase tuples, including conditional identity |
| repr and class/method registration | Native private format class identity and existing module ownership |
| Original documentation examples | All 47 actual frozen-extension examples execute |

The existing count_operations CLI now consumes this authoritative projection.
Its output categories and debug-print row retain their original format.
Simulation exposes a counting observer on the existing execution loop and sample
entry. The native caller owns packed state and cumulative counters separately.
This replaces the original combined simulator member ownership at the existing
Compiler/Simulation boundary. Native repr uses PhysicalOperationStatistics.

## Accepted execution and remaining boundaries

The frozen fixture verifies 30 transitive source/header blobs. The two original
test source blobs and all three original counting/resource/mapping tests reuse
the intake receipt. Seven complete static observations, 90 range queries,
10 aggregate snapshots and 1,144 individual lane snapshots agree with the actual
locked extension. Every exact rotation level is exercised both statically and
through the real counted execution loop. The corpus regenerates byte-exactly.
Repeated lane snapshots use an index table to retain every observation without
repeating identical report data. Original files remain unchanged.

The complete affected regression passes 380 items, integration passes 121 and
strict typing passes all seven touched Python files. Statistics, lane tally and
shared execution loop each have 100% measured statement/branch coverage.
Black, Ruff, scope, architecture, IR and protected API checks pass. Dependency
policy retains 720 existing findings, with zero findings in touched files.

The observed and unobserved execution paths agree on packed wires, classical
feedback and exact 128-bit phase. All four small two-measurement erasure histories
match their complete complex amplitudes and both work wires return to zero.
The normal DynamicCircuit.append(op) path independently covers all four histories
with the same states. Static primitive histograms, actual executed counts and
analytical estimates have separate fields in the acceptance receipt. Instrumentation
adds zero physical operations, measurements and workspace. This counting source
declares no independent analytical quantum estimate.

Synthetic uint64 capacity checks exercise wrapping explicitly. They carry
unsigned-arithmetic evidence, separately from finite source execution observations.
The native helper boundaries accept owned Python values, while source binding
conversion remains the external fixture boundary. Full interactive simulator
wrappers and their remaining methods retain their own parent acceptance scope.
Pending Proposals 116–119 remain recorded in the main ledger.
