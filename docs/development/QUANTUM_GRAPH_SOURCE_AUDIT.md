# Quantum graph source integration audit

## Frozen files and native ownership

This audit covers the complete Qualtran v0.7.0 graph metadata file and its original tests at revision `bcb17bb25360ed68ce3bc084645db3e929799389`. Both blobs are verified against the frozen inventory.

| Parent | File | Git blob |
|---|---|---|
| Q-0165 | `qualtran/_infra/quantum_graph.py` | `923ddb3b6c79606ae5bd5c9ed23d22675b2028d9` |
| Q-0166 | `qualtran/_infra/quantum_graph_test.py` | `c4d12d29788d07975a139454622917c061bfd0f9` |

Proposal 103 assigns graph authority to existing Program SSA. Core owns immutable definitions, operands, results, explicit return values, Register and RegisterBinding. Compiler owns building, validation, call traversal, scheduling and physical binding. Drawer owns endpoint labels and displayed connections. Simulation owns numerical execution. This is a `single_device_fast_path` source audit. This batch reuses the existing production implementations for the behaviors below.

## Complete source definition mapping

The source has 17 class/function/method occurrences. The two singleton values and installed constructor guard are accounted for separately.

| Source definition | Native representation and consumer | Evidence |
|---|---|---|
| `BloqInstance` | Selected FunctionRecord plus call ordinal within its owning block | Repeated actual calls share a definition and produce distinct SSA outputs. |
| `BloqInstance.__str__` | Drawer function name and occurrence label | Full source debug text matches after replacing source instance names with native callee labels. |
| `BloqInstance.bloq_is` | Compiler selectors receive Operation and its callee in the existing definition table | Definition identity and actual call traversal distinguish components from graph boundaries. |
| `BloqInstance.__hash__` | Immutable FunctionRecord, call ordinal and ValueRef keys | Equal instances resolve to the same dictionary entry; distinct definitions remain distinct at the same ordinal. |
| `DanglingT` | Block arguments and explicit returns; Drawer producer/consumer boundary fields | Actual input and output edges remain distinct. |
| `DanglingT.__init__` | Existing Program boundary ownership | Boundary position is supplied by the owning block and edge direction. |
| `DanglingT.__repr__` | Drawer boundary labels | Exact source endpoint text is retained by the existing debug renderer. |
| `DanglingT.bloq_is` | Component selectors iterate actual calls only | Boundaries are excluded from call traversal. |
| `_to_tuple` | Core RegisterBinding constructor | Python integer conversion, tuple retention and source bounds behavior reused from existing source-backed binding tests. |
| `Soquet` | RegisterBinding over existing ValueRef | Scalar, shaped, empty and symbolic-width metadata keep their index and logical ownership. |
| `Soquet._check_idx` | Core RegisterBinding index validation | Rank precedes bounds; negative indices and symbolic-shape refusal reuse accepted tests. |
| `Soquet.pretty` | Drawer `_port_labels` | Original `x` and `y[5, 0]` labels compared. |
| `Soquet.__str__` | Drawer endpoint rendering | Full graph diagnostics compare each producer, consumer, register and element index. |
| `_singleton_error` | Canonical block-boundary representation | Original constructor refusal passes; native boundaries use the contextual input/output positions in the authoritative Program. |
| `Connection` | SSA operand/result links; modeled WidthConnection and displayed RegisterConnection projections | Complete connectivity is compared against the source graph. |
| `Connection.shape` | Register bitsize plus existing typed verifier and width projections | Equal-width storage, zero width, symbolic width and mismatched typed return checks pass. |
| `Connection.__str__` | Drawer component debug text | Source `left -> right` labels and complete graph text agree. |

`LeftDangle` and `RightDangle` map to the block input and output roles. A through value can appear in both roles without erasing that distinction. The native displayed edge stores the role in its producer or consumer field. Source singleton construction and native graph-boundary representation have separate APIs under Proposal 103.

The frozen instance hash intentionally depends only on its ordinal, so different source definitions can collide. Native immutable record hashing follows its own stored fields. Acceptance concerns equality, dictionary lookup and distinct occurrence semantics. Numeric hashes follow the stored fields of each representation. Source class names and method signatures remain external SDK vocabulary.

## Original tests and direct dependencies

All five original tests execute unchanged. Native counterparts in `tests/unit/test_quantum_graph_source.py` cover input/output boundary distinction and dictionary lookup, scalar endpoint side/index/label, the exact 10-by-2 indexed endpoint and its rank errors, immutable equality, and definition-plus-occurrence identity. Additional checks exercise all members that the original tests omit, including component type classification, connection equality and text, zero/symbolic width storage and width mismatch diagnostics.

Existing accepted evidence is reused through `test_register_bindings.py`, `test_register_component_calls.py`, `test_component_program_builder.py`, `test_component_graph.py`, `test_component_debug_text.py`, `test_component_physical_schedule.py`, `test_register_physical_binding.py`, `test_drawer_component_score.py`, `test_drawer_register_graph.py`, `test_program_registers.py` and `test_register_tensor_indices.py`. This covers scalar and array bindings, fresh call successors, single consumption, empty words, explicit returns, exact graph endpoints, source traversal order, source tensor index mapping and actual measured Add/QROM consumers.

## Validation and representation boundaries

Source Connection can store an invalid width pair and raises when its cached shape is requested. Native finite Programs validate the producer/consumer type relation at the existing verifier boundary and report `program.register_type`. The timing and diagnostic vocabulary differ because there is no public foreign Connection object in the native architecture. The test checks both owning boundaries explicitly.

A symbolic connection width remains metadata in Core and the source analytical width representation. Executable width admission is independently validated by Compiler. Source graph metadata supplies no physical implementation for an arbitrary child. Broader classical-word admission remains with Q-0158 and pending Proposal 108. Optional SDK zero-width inference remains with Q-0163. These consumer boundaries do not remove indexed or zero-width identities from the native graph.

The source generic graph has no gate stream or measurement of its own. Numerical evidence therefore uses actual graph consumers. Existing exhaustive Add/QROM measurement histories and clean-work checks remain authoritative. The focused two-X consumer compares its complete complex operator, a data qubit entangled with a spectator, an idle zero work wire, and ordinary `Circuit.append` execution. Its two calls preserve distinct SSA successors and three ordered graph connections.

## Resource accounting

| Path | Emitted primitives | Stored graph | Source analytical model |
|---|---|---|---|
| Pure metadata construction | 0 | Register/endpoint/connection metadata | Physical decomposition inapplicable |
| Two-X actual consumer | 2 X | 2 shared calls, 3 ordered connections | 2 Clifford, 0 T, 0 measurement |
| Add/QROM measured consumers | Existing exact primitive and feedback evidence reused | Existing typed call and lifetime evidence | Existing source QEC and width models remain separate |

The two-X example allocates no workspace and performs no measurement. Its unused supplied work wire stays zero. Measurement branches are inapplicable to that example. Measured consumers use their existing exhaustive branch tests.

## Reproduction

Run the complete original test file and the native source audit with the locked reference environment. Run the concentrated graph/binding/Drawer/tensor consumers listed above after preimporting Torch. Environment, source blobs, logs and generated receipts are stored under `../ftqc-reference-env/g2-quantum-graph-*`. The default whole-repository baseline from `e855deac8` selected 74,391 cases and timed out after 480 seconds with existing example/composition failures. This audit reuses that recorded limitation. Its changes are limited to tests and documentation.

The final combined regression passes 1,571 cases with eight existing pydot deprecation warnings and no skips. The scoped default smoke/unit check passes 87 cases. The two changed Python files pass strict typing, Black and Ruff. Team scope, architecture and IR checks pass. Dependency scanning reports zero findings on changed paths and 688 whole-repository findings. The graph focused run passes ten native cases and all five original cases, with two original helper-class collection warnings. The Register/Signature focused run passes nineteen native cases and all eighteen original parametrized cases.
