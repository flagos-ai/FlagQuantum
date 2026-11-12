# Register musical score notebook

The frozen source is Qualtran v0.7.0 `qualtran/drawing/musical_score.ipynb`,
commit `bcb17bb25360ed68ce3bc084645db3e929799389`, Git blob
`7205fa53b168a8b0de5429345dc25d2081979446`.
The native asset is `examples/notebooks/register_musical_score.ipynb`.
Both contain seven cells, comprising three code cells and four Markdown cells.
The native notebook stores unique cell IDs, Python kernel metadata and empty
outputs. Its text retains upstream attribution and the Apache-2.0 license.

## Ownership and cell disposition

This is a `single_device_fast_path` example. Compiler's existing MultiAnd stages
supply the layout through `multi_and_register_score`. Drawer retains ownership
of layout, Graphviz, Matplotlib, JSON and offline HTML. This batch adds no
production implementation, graph representation or numerical provider.

| Source cell | Native responsibility |
| --- | --- |
| 0 | Explain dependency graphs and left-to-right register lifetimes |
| 1 | Explain the four-control AND graph |
| 2 | Build the shared ladder for `(1, 1, 0, 1)` and display its Graphviz SVG |
| 3 | Explain score-data extraction, coordinate ownership and rendering |
| 4 | Extract score data, display Matplotlib figure and close the figure |
| 5 | Explain JSON export and the Javascript consumer |
| 6 | Write `musical_score_example.json` and a self-contained native HTML viewer |

The original three code cells execute unchanged in a temporary output directory,
including the source's relative JSON filename. The frozen example JSON remains
untouched. Both native notebook assets also execute in independent Jupyter
kernels that reject imports of Qualtran, Quimb and Cirq. All three native score
cells execute sequentially, display one SVG and one PNG, and write JSON/HTML
whose embedded sequential data agree. Execution leaves tracked notebook bytes
unchanged. Kernel and export directories belong to the test caller.

## Comparison and adaptations

The full source/native JSON comparison retains maximum coordinates, horizontal
lifetimes, vertical connections and every symbol attribute. Only port `ident`
strings differ, following existing native logical-lane identity. Port identities
remain unique and all remaining port records match by coordinate. JSON formatting
and key order are outside this comparison.

The same negative-control example produces pixel-identical Agg figures under
matched font and DPI settings. Axes limits also agree. The saved native score
was visually inspected, including the third control's open circle.
Graphviz uses the existing native instance graph exporter, with native IDs and
boundary labels. Graphviz pixel identity is outside this acceptance.
The native self-contained HTML page replaces the source's separate D3 page and
script. Javascript animation equivalence remains open under Q-0744/Q-0746.

## Physical gates and analytical model

The existing MultiAnd tests compare primitive sequences, operands, measurement
feedback and all eight observed erasure branches. A complex superposition returns
to its initial amplitudes with all three caller-owned prefix/target work qubits
zero. The ordinary user path remains `circuit.append(op)`. This batch changes
presentation only.

| Account | Locked negative-control example |
| --- | --- |
| Actual forward primitives | H6, T6, Tdg6, CX18, S3, X2 |
| Actual erasure primitives | H3, measure3, conditional CZ3, reset3, X2 |
| Actual quantum storage | Four controls and three caller-owned output/work wires |
| Source forward QEC model | AND3 and Clifford2, equivalent T count 12 |
| Source adjoint QEC model | Clifford5 and measurement3 |
| Source peak model | Seven qubits |
| Added by drawing and export | Zero gates, work qubits, measurements or feedback |

The static AND-level Clifford account and emitted primitive Clifford account
remain separate. Each phase's two X gates preserve the negative control.

## Local acceptance

Three focused tests pass, including the existing gate/branch acceptance. Two
independent native notebook kernel cases pass. The unchanged source notebook
also passes in a real reference kernel. The Drawer and graph-notebook regression
passes 212 tests with eight existing parser warnings. CPU/tensor-network
regression passes 84 tests with two existing warnings. No cases skip.
Strict typing of the two changed Python test files, Black, Ruff, team scope,
architecture, multi-level IR and protected public API checks pass. The source
notebook blob is verified before comparison. No production coverage claim is
added because production owners are unchanged.

Run with Graphviz and notebook dependencies in the locked reference environment.

```bash
python -m pytest tests/unit/test_musical_score_notebook.py \
    tests/unit/test_drawer_multi_and_score.py \
    tests/integration/test_register_graph_notebook.py -q
```

Evidence prefix is `g2-score-notebook-` in the external reference workspace.
Q-0745 is accepted as a complete notebook asset. Generic drawing classes,
source-wide symbol coverage and Javascript animation remain separate open items.
