# Call graph protocol source audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Source file | Frozen Git blob | Native disposition |
| --- | --- | --- |
| `resource_counting/_call_graph.py` | `8c1a6debd8c8a2e06d3a082dc6b6cd6656e4f522` | Compiler owns direct declarations, symbol allocation, traversal and weighted counts. Drawer owns the debug text |
| `resource_counting/_call_graph_test.py` | `af540e50c004fa3d7a2eaa22a221060dea42e737` | Eight original non-notebook cases run unchanged. Native assertions cover logarithmic counts, declaration and fallback, shared paths, rewrites and duplicate calls |
| `resource_counting/call_graph.ipynb` | `8c6d5e894f2b0ba1729a74bde77dfbb73aa61e7f` | Native `call_graph_protocol.ipynb` covers all 23 cells by topic and executes all 10 code cells with five SVGs |

## Source definition map

| Source definition | Native owner |
| --- | --- |
| `big_O` | `asymptotic_order` uses infinity for the free symbols and preserves constants |
| `SympySymbolAllocator` | Per-query `SymbolAllocator` uses an independent counter per prefix |
| `build_cbloq_call_graph` | Stored typed Program occurrences feed `program_component_call_graph` |
| `_generalize_callees` | One normalization path rewrites, omits and combines direct calls by their first appearance |
| `get_bloq_callee_counts` | `component_callee_counts` handles ordered mappings and iterable declarations with an optional allocator and strict capability errors |
| `_build_call_graph`, `get_bloq_call_graph` | Shared Compiler traversal accepts static maps or callbacks and honors the first-visit cutoff |
| `_compute_sigma` | The existing topological accumulation sums multiplicities over all paths to each terminal node |
| `format_call_graph_debug_text` | Drawer formats topological generations in parent and child label order |

A missing declaration or a native capability refusal can produce a
terminal node. The direct-count query can propagate that refusal when
requested. The native graph rejects cycles and invalid depth arguments.
It stores weighted edges in the existing graph record. Source identities
used in tests are converted at the comparison boundary.

The notebook reproduces the all-positive and mixed conjunction examples,
generalized activation symbols, a symbolic two-AND-per-element and
five-CNOT declaration, the stored three-CNOT SWAP, and agreement between
a declarative model and a ladder decomposition under the same rewrite.
The source prose says the explicit Clifford count falls after generalizing
control values. The pinned graph retains six X calls in the mixed-polarity
example before and after the rewrite. Both the model and the native
notebook show that behavior.

| Example | Source model | Native primitive instructions |
| --- | --- | --- |
| Six positive controls | AND5, logical width 11 | Graph query emits zero instructions |
| Symbolic function | AND `2*n`, CNOT5 | Declaration emits zero instructions |
| Stored SWAP | CNOT3, Clifford3 | CX3 in source gate and operand order |

The stored SWAP reaches `circuit.append(op)` and its full complex state
and output-wire binding match the source permutation. It uses no private
work or measurements. Existing predicate tests cover measurement feedback
and all small erasure histories under the same resource owners.

The source and native notebooks were run in independent kernels and their
five SVGs inspected. The drawing-specific `resource_call_graphs.ipynb`
remains the separate annotation and DOT tutorial.
