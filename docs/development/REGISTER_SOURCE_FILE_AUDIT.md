# Register and Signature complete source audit

## Frozen source and scope

The complete frozen source is Qualtran v0.7.0 revision `bcb17bb25360ed68ce3bc084645db3e929799389`. Source identities are checked in the executable audit. The batch reuses the existing production implementations and verifies their remaining whole-file obligations.

| Parent | File | Git blob |
|---|---|---|
| Q-0167 | `qualtran/_infra/registers.py` | `cfc9191ff398687cd0f86ea3a38ee33a58c25cee` |
| Q-0168 | `qualtran/_infra/registers_test.py` | `7da20da8624fee7203e9d4a7354091d47f0a3e1c` |

Core owns `program/registers.py`, `program/signatures.py`, the prevalidated `register_types.py` factories and shared symbolic width arithmetic. Compiler owns binding collections, typed graph construction and physical placement. Ecosystem owns SDK named-qubit arrays. Simulation owns value encoding and numerical execution. The path is classified `single_device_fast_path`.

## Complete definition and dependency mapping

The inventory has 36 class/function/method occurrences, including two overload declarations and their implementation.

| Source definition | Line | Native owner and evidence |
|---|---|---|
| `Side` | 31 | RegisterSide LEFT/RIGHT/THRU values and source adjoint/index tests. |
| `Register` | 53 | Core immutable Register metadata; original scalar/classical/multidimensional bodies. |
| `Register.__attrs_post_init__` | 77 | Native RegisterType nominal validation; source type spelling translated at the owning boundary. |
| `Register.is_symbolic` | 81 | Shared Core is_symbolic_value and descriptor protocol. |
| `Register.shape_symbolic` | 85 | Exact declared scalar/array/symbolic extents. |
| `Register.shape` | 89 | Symbolic shape refusal and original conversion timing. |
| `Register.bitsize` | 95 | Combined quantum/classical bit count. |
| `Register.all_idxs` | 98 | Original row-major product, empty arrays and scalar zero-width elements. |
| `Register.total_bits` | 102 | Exact bitsize-times-shape product and scalar return type. |
| `Register.total_qubits` | 109 | Quantum width independent of classical storage. |
| `Register.total_cbits` | 117 | Classical width independent of quantum storage. |
| `Register.adjoint` | 125 | THRU identity and LEFT/RIGHT exchange, unknown-side refusal. |
| `_dedupe` | 136 | Source-ordered per-side duplicate checking after tuple consumption. |
| `Signature` | 147 | Existing Core immutable ordered register collection. |
| `Signature.__init__` | 179 | Tuple snapshot, per-side names and shared-name opposite sides. |
| `Signature.build` | 185 | Core bit/opaque factories and original truth-value filter. |
| `Signature.build_from_dtypes` | 210 | Quantum-width filtering, including explicit omission of classical-only types. |
| `Signature.thru_registers_only` | 235 | Existing cached property with empty/THRU/directional signatures. |
| `Signature.lefts` | 241 | Ordered input and THRU registers. |
| `Signature.rights` | 245 | Ordered output and THRU registers. |
| `Signature.get_left` | 249 | Exact input name lookup and KeyError. |
| `Signature.get_right` | 253 | Exact output name lookup and KeyError. |
| `Signature.groups` | 257 | Insertion-ordered grouping of same-name opposite-side registers. |
| `Signature.adjoint` | 268 | Native Register.adjoint applied in original declaration order. |
| `Signature.n_qubits` | 272 | Maximum of independent left/right quantum sums. |
| `Signature.n_cbits` | 285 | Maximum of independent left/right classical sums. |
| `Signature.n_bits` | 296 | Maximum of independent left/right total sums. |
| `Signature.__repr__` | 311 | Existing native Signature tuple display. |
| `Signature.__getitem__` | 315 | Integer and slice overloads plus exact tuple indexing failures. |
| `Signature.__getitem__` | 319 | Integer and slice overloads plus exact tuple indexing failures. |
| `Signature.__getitem__` | 322 | Integer and slice overloads plus exact tuple indexing failures. |
| `Signature.__contains__` | 325 | Stored-register value membership. |
| `Signature.__iter__` | 328 | Declaration-order iteration. |
| `Signature.__len__` | 331 | Exact stored tuple length. |
| `Signature.__hash__` | 334 | Hash of the immutable stored register tuple. |
| `Signature.__eq__` | 337 | Original foreign _registers attribute behavior and uncoerced equality result. |

All declared fields are preserved by the native metadata owners. Register retains name, descriptor, symbolic shape and direction. Signature retains the declaration tuple and its independent left/right name maps. Attributes and cached properties are owned by these existing classes. The source imports QBit/QAny/QCDType for metadata construction, nominal identity and bit counts. Their native equivalents are the existing bit/opaque factories and RegisterType declaration. Full value codecs have their own Simulation scope.

## Every original test body

`tests/unit/test_register_source_scenarios.py` executes sixteen original zero-argument test function code objects using isolated globals that select the native Register, Signature, factories, symbolic classifier and named-qubit owner. The function bytecode and assertions are the frozen originals. The source module globals and files retain their original objects. This is native dependency substitution evidence, recorded separately from the original source suite.

The test-only bounded constructor bridge translates the original keyword names `bitsize` and `iteration_length` to the existing native width/bound factory. Native bounded metadata stores the iteration bound in descriptor parameters. The original two selection-indexing parameter cases therefore have explicit native counterparts that read that owned parameter, then compare every ravel/unravel coordinate and the product of bounds. No parallel dtype class is introduced.

| Original function | Native counterpart or reused evidence |
|---|---|
| `test_register` | Original frozen function body with native dependency globals. |
| `test_classical_register` | Original frozen function body with native dependency globals. |
| `test_multidim_register` | Original frozen function body with native dependency globals. |
| `test_selection_registers_indexing` | Both original parameter tuples and every bounded coordinate, using native bound metadata. |
| `test_selection_registers_consistent` | Original frozen function body with native dependency globals. |
| `test_registers_getitem_raises` | Original frozen function body with native dependency globals. |
| `test_signature` | Original frozen function body with native dependency globals. |
| `test_get_named_qubits` | Original frozen function body with native dependency globals. |
| `test_signature_symbolic` | Original frozen function body with native dependency globals. |
| `test_partial_classical_signature_n_qubits` | Original frozen function body with native dependency globals. |
| `test_signature_build` | Original frozen function body with native dependency globals. |
| `test_and_regs` | Original frozen function body with native dependency globals. |
| `test_agg_split` | Original frozen function body with native dependency globals. |
| `test_get_named_qubits_multidim` | Original frozen function body with native dependency globals. |
| `test_duplicate_names` | Original frozen function body with native dependency globals. |
| `test_dtypes_converter` | Original frozen function body with native dependency globals. |
| `test_is_symbolic` | Original frozen function body with native dependency globals. |

The original seventeen functions yield eighteen parametrized source cases. Sixteen preserved bodies plus the two adapted indexing cases and one source identity check yield nineteen native audit cases. The combined focused run passes all 37 cases without skips or warnings.

Existing `test_register_metadata.py`, `test_register_signature.py` and `test_register_type_constructors.py` retain the wider boundary evidence. They cover scalar type and exception behavior, generators, unknown sides, symbolic versus concrete shapes, negative/zero dimensions, classical storage, bounded types, independent side peaks, equality and hashing, lazy convenience constructors, source filtering order and actual Add/QROM signature replacement. Q-0165/Q-0166 now supply the whole graph/endpoint dependency audit in `QUANTUM_GRAPH_SOURCE_AUDIT.md`. The complete named-register tutorial and SDK gate tests are recorded in `REGISTER_GATE_SOURCE_AUDIT.md`.

## Numerical and resource consumers

The two metadata owners emit zero gates. Their numerical obligations are proven at actual graph consumers. The concentrated batch reuses Add/QROM exact primitive streams and complete measurement histories, full complex amplitudes, clean work, typed calls, tensor indices, named placement and the normal `circuit.append(op)` entry. Model width and primitive counts stay at their owning consumers.

For the focused two-X graph, the emitted stream is two X primitives, two stored calls and three graph connections. The frozen source QEC model gives two Clifford gates, zero T and zero measurement. The entangled spectator is preserved and the unused work wire stays zero. The register tutorial separately emits six CX for ten split/call/join graph operations and a source model of six Cliffords.

## Consumer boundaries and evidence reuse

A declared symbolic width or a classical word retains metadata identity. Physical width admission and feedback execution follow the existing Compiler contract. Wider classical Program admission remains pending Proposal 108 under Q-0158. SDK zero-width inference remains under Q-0163. These are distinct consumer obligations from the complete Register/Signature metadata source.

Evidence resides in `../ftqc-reference-env/g2-register-source-*` and the combined `g2-quantum-graph-*` receipts. The original reference suite, substituted dependency suite, concentrated native graph regression and strict checks are recorded separately. The prior whole-repository default tier from `e855deac8` timed out after 480 seconds with example/composition failures. That baseline is reused for this tests-and-documentation batch.

The final combined regression passes 1,571 cases with eight existing pydot deprecation warnings and no skips. The scoped default smoke/unit check passes 87 cases. The two changed Python files pass strict typing, Black and Ruff. Team scope, architecture and IR checks pass. Dependency scanning reports zero findings on changed paths and 688 whole-repository findings. The graph focused run passes ten native cases and all five original cases, with two original helper-class collection warnings. The Register/Signature focused run passes nineteen native cases and all eighteen original parametrized cases.
