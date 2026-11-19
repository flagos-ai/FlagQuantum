# Component connection text

`flagquantum.drawer.component_debug_text.format_component_debug_text` owns the
text presentation previously absent from the Q-0156 mapping. It follows frozen
Qualtran v0.7.0 `CompositeBloq._debug_binst` and `debug_text`, including
twenty-character generation separators, input and output edge duplication,
adjacency grouping, connection insertion order and multidimensional indices.
Native function names and `name#instance` labels replace source Bloq names and
`name<instance>` labels. An optional `instance_ids` tuple accepts the labels
already returned by selective expansion. It uses the existing label validator.

The existing component score owns graph validation and generation assignment.
The renderer reads the original Program SSA for endpoint identity. It introduces
no second graph representation, gate decomposition, numerical execution or
workspace allocation. The same finite, unconditional, unparameterized typed-call
profile as the existing score is supported. Symbolic Program admission remains
with the open Q-0149 scope. The source's ordering of disconnected zero-port nodes
depends on frozenset iteration; native stored-call order remains deterministic.

The native regression compares the complete text against the unmodified source
after translating only component names and instance delimiters. Cases include
the full split / parallel-X / join graph, scalar and multidimensional registers,
zero widths, empty shapes, reversed bit connections, no-call identity, invalid
entry and labels retained after expansion. The original graph retains 14 printed
edges in three generations. Source and native text preserve both occurrences of
every internal connection.

An appended X Program remains unchanged before and after rendering and produces
the same physical instruction. This presentation function adds zero primitives,
measurements and workspace. Existing Program import and graph regressions supply
the full complex-amplitude and work-zero witnesses for the unchanged executable
path. Resource-model estimates are not recomputed by a text renderer.

Initial parity checks caught output edges being emitted in producer-port order
instead of source connection order, and missing spaces in multidimensional
indices. Both were fixed in the renderer. A test's incorrect 16-edge expectation
was corrected to the source's actual 14 edges. These failures did not modify the
frozen reference. This completes the text-rendering dependency, while Q-0156 as
a whole remains open for its remaining conversion and descriptor boundaries.

All 16 focused cases pass. Concentrated graph, graphviz and Program-import
regression passes 55 cases, with eight upstream rendering deprecation warnings
and no skips. The renderer's 41 statements and 10 branches have 100% measured
coverage. Strict typing, Black, Ruff, Compiler scope and architecture checks
pass. The dependency scan retains 706 existing findings with no findings in
the changed Python files.
