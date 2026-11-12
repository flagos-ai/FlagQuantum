# Musical score source audit

This audit covers Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Frozen source | Git blob |
| --- | --- |
| `qualtran/drawing/musical_score.py` | `3863bb66cdba17c4654657e275a8485d94b8da75` |
| `qualtran/drawing/musical_score_test.py` | `db35be86a8fc4c1f3dc25b4154a89776ec519565` |
| `qualtran/drawing/musical_score.ipynb` | `7205fa53b168a8b0de5429345dc25d2081979446` |
| `qualtran/drawing/musical_score.html` | `8731aa90a752d4fd252ea931f0527266da0c3934` |
| `qualtran/drawing/musical_score.js` | `ead9b24fc77b0a772e05f18280a734ddf65ee3a1` |

The native owners are `drawer/register_score.py`, `component_score.py`,
`register_score_data.py`, `register_score_mpl.py` and the shared heap owner
`register_lanes.py`. Compiler owns the selected
component graph and validation. Drawer owns display coordinates, lifetime
projection, symbols and serialization. Simulation continues to own numerical
execution. This is a `single_device_fast_path` audit and test batch.

## Complete definition disposition

The table includes every top-level source function and class, with all methods.
The external AST evidence also records line numbers, nested helpers and imports.
Existing native records remain the shared representation.

| Source definition and methods | Existing native owner and disposition |
| --- | --- |
| `RegPosition.json_dict` | `RegisterColumn` stores sequential and generation coordinates; `ScorePort` stores exported coordinates. Plain dictionaries provide serialization. |
| `HLineFlavor.from_qcdtype` | `register_score_data(classical=...)` explicitly marks classical display identities. Automatic foreign dtype inspection and its unknown-type fallback are outside native admission. |
| `HLine.json_dict` | `RegisterLine` stores lifetimes; `ScoreHorizontalLine` serializes completed lines with source flavor spelling. Partial foreign line records remain internal to the source manager. |
| `LineManager.__init__`, `new_y` | `RegisterLaneManager` owns the shared minimum-lane heap consumed by `layout_register_score`. Native capacity grows with the graph. The source defaults to 100 lines. |
| `LineManager.reserve_n`, `unreserve`, `maybe_reserve` | Native `RegisterLaneManager` now implements ordered reservations and the `maybe_reserve` subclass hook; Core grouping preserves once-per-register predicate cadence and per-element hooks. See REGISTER_SCORE_RESERVATIONS.md for finite-capacity parity and failure adaptations. |
| `LineManager.new`, `finish_hline`, `free` | Existing native layout allocates fresh RIGHT lanes before freeing LEFT lanes and closes lifetimes at consumption or final output. `_ports` flattens shaped register elements in row-major order. Native reservation callbacks now cover these boundaries. Source partial-HLine inspection and new-coordinate overrides remain open. |
| `_get_in_vals` | `component_register_score` resolves existing Core SSA values through its `known` mapping and `_ports` iterator. No foreign Soquet map is introduced. |
| `_update_assign_from_vals` | Native component projection maps THRU outputs to predecessor lifetimes and creates fresh RIGHT display identities. Core validation owns array signatures. Foreign array-value shape diagnostics remain open. |
| `_binst_assign_line`, nested `_in_vals` | Native component projection consumes the selected call's operands/results, retains THRU identity and retires LEFT lifetimes through the shared layout. |
| `_cbloq_musical_score`, nested `_f_vals` | `component_register_score` projects the selected Core graph; `layout_register_score` owns dependency generations and output lifetimes. Existing `register_program_score` separately projects verified register constructions. Foreign graph-manager replacement remains open. |
| `WireSymbol.draw`, `adjoint`, `json_dict` | Existing `SymbolClass` and attribute dictionaries supply the six glyph classes. `_draw_port` owns drawing. `adjoint_wire_symbol` now transforms all six admitted record classes. Arbitrary foreign symbol subclasses remain open. |
| `_text_adjoint` | `adjoint_wire_symbol` now preserves the exact source dagger add/remove and `strip` behavior. |
| `TextBox.draw`, `adjoint` | `_draw_port` draws the round box. `adjoint_wire_symbol` supplies dagger transformation through the existing callback. |
| `Text.draw`, `adjoint` | `_draw_text` preserves text and font size. `adjoint_wire_symbol` supplies dagger transformation through the existing callback. |
| `RarrowTextBox.draw`, `adjoint` | `_draw_port` draws the right arrow. Native LEFT ports select it explicitly. `adjoint_wire_symbol` switches directional boxes and retains their text. |
| `LarrowTextBox.draw`, `adjoint` | `_draw_port` draws the left arrow. Native RIGHT ports select it explicitly. `adjoint_wire_symbol` switches directional boxes and retains their text. |
| `Circle.draw` | `_draw_port` preserves filled/open circles, radius and color. |
| `ModPlus.draw` | `_draw_port` preserves the addition marker, font and transparent box. |
| `directional_text_box` | `_port_symbol` selects through/left/right box styles for admitted native ports. Source enum error behavior and a standalone foreign helper remain outside that surface. |
| `_soq_to_symb` | `register_score_data` builds native boundary labels and accepts the existing `wire_symbol` callback for component ports and captions. Callback ownership remains presentation-only. |
| `SoqData.json_dict` | `ScorePort` stores symbol, coordinates and identity together. |
| `VLine.json_dict` | `ScoreVerticalLine` and `ScoreLabel` retain connector geometry, text and font size. |
| `MusicalScoreData.json_dict` | Existing typed `MusicalScoreData` is a plain mapping with the source JSON field names. |
| `_make_ident` | Existing native identity strings name logical lane lifetimes and the original operation ordinal. Foreign instance/Soquet identity spelling is explicitly adapted. |
| `get_musical_score_data` | Native component projection, layout and score-data extraction form the equivalent admitted call chain. Native lane-reservation managers are supported. Generic foreign Bloqs, foreign line managers and source-wide wire-symbol protocols remain open. |
| `draw_musical_score` | `draw_register_score` preserves sequential geometry, sizing, colors, glyph order and axes. Seven direct glyph render cases supplement existing whole-score pixel comparisons. |
| `MusicalScoreEncoder.default` | `export_register_score_json` serializes the existing plain records through the standard JSON encoder. Foreign object type dispatch is owned by the reference test boundary. |
| `dump_musical_score` | Existing JSON export plus caller-owned `Path.write_text`, demonstrated by the accepted native notebook. File ownership stays with the caller. |

Source imports of `attrs`, NumPy arrays, NetworkX graphs and foreign
Bloq/Register/Soquet types stop at reference tests. Native production continues
to use existing dataclasses, Core signatures/values and Python graph/heap
utilities. Matplotlib remains an optional rendering dependency. No parallel graph
IR, foreign runtime adapter, symbol hierarchy or new production dependency is
added by this audit. Existing native owners retain source copyright notices.

The HTML template and all thirteen JavaScript functions now have a complete
ownership table in REGISTER_SCORE_ANIMATION.md. The existing offline renderer
implements keyed enter/update/exit with the source's 750 ms cubic-in-out timing.
Real Chrome comparisons execute the unchanged source script and pinned test-only
D3, explicitly aligning coordinate ranges. The native URL form now follows
the source HTTP/JSON/error-panel chain and text-glyph click diagnostics. Native
preflight validation, newest-request ownership and imported sequential-view
selection have separate acceptance. Fixed style now adds source axes, margins, sizing,
font inheritance and lightblue lines through that same owner. Twelve new browser
cases compare default ranges directly, including original MultiAnd JSON, two
pixel ratios and HTTP reflow. Automatic native domains and the initial Adaptive
style have explicit dispositions in REGISTER_SCORE_ANIMATION.md. Arbitrary
manual domains and complete page-boundary acceptance remain open.
Q-0744 and Q-0746 therefore remain open. The ordinary append and numerical paths
continue through their existing owners.

## Complete original test matrix

| Original test | Unchanged source execution | Native acceptance |
| --- | --- | --- |
| `test_dump_json` | The original HLine default assertion and MultiAnd dump run directly. | Its exact original code object runs again with test-only construction/record inspection views backed by native layout/data/JSON owners. Full geometry, lines and non-identity port records are compared after export. |
| `test_notebook` | Runs the source `execute_notebook('musical_score')` helper in a real reference kernel. CWD is an isolated output directory, preserving the frozen notebook and example JSON. | The accepted Q-0745 notebook executes all three code cells in an independent kernel with Qualtran, Quimb and Cirq imports refused. SVG, PNG, JSON and embedded HTML data are checked by `test_register_graph_notebook.py`. |

The native record inspection views exist only in the test. They introduce no
production foreign protocol. The original test code and assertions are
unchanged. The source notebook's 600-second helper is enclosed in a bounded
100-second process-group verification run.

## Render and boundary evidence

Seven glyph cases compare exact Agg pixels for TextBox, Text with a nondefault
font size, both directional boxes, filled and open circles, and ModPlus. Each
uses source-produced JSON directly through the native renderer, including a
nonempty connector caption. Figures are closed in all cases.

Two additional cases record meaningful representation differences. The source
rejects an empty graph when taking the maximum of an empty position sequence.
Native empty metadata produces no ports or lines. The source default manager
exhausts its 100-line heap for 101 independent input registers. With an explicitly
sized 101-line source manager, native lifetimes, maximum lane and 202 boundary
ports agree. Arbitrary reservation hooks remain outside that comparison.

The existing MultiAnd acceptance continues to verify gate/operand/feedback
parity, complex superposition restoration across eight observed erasure branches,
three work wires returning to zero, and ordinary `circuit.append(op)` execution.

| Accounting layer | Existing negative-control MultiAnd result |
| --- | --- |
| Actual forward instructions | H6, T6, Tdg6, CX18, S3, X2 |
| Actual erasure instructions | H3, measurement3, conditional CZ3, reset3, X2 |
| Source forward model | AND3 and Clifford2, equivalent T count 12 |
| Source adjoint model | Clifford5 and measurement3 |
| Storage | Four controls and three caller-owned output/work wires; source peak seven |
| Added by this audit | Zero primitive gates, work allocation, measurements or feedback |

No production execution path changes in this batch. Physical primitive counts
and source analytical models retain their distinct accounting layers.

## Acceptance and next dependencies

Q-0749 closes for the complete two-function original test file after both real
source tests and the native equivalents pass. Q-0747 remains open for the
general foreign, partial-line inspection and custom-coordinate protocol boundaries listed
above. Native reservation and subclass hooks are documented in
[REGISTER_SCORE_RESERVATIONS.md](REGISTER_SCORE_RESERVATIONS.md). The accepted Q-0745 notebook and Q-0749 tests do not imply those missing
implementation extensions are available. The builtin glyph adjoint follow-up is documented in [REGISTER_SCORE_SYMBOL_ADJOINT.md](REGISTER_SCORE_SYMBOL_ADJOINT.md). Q-0744/Q-0746 Javascript animation also
remains separately open.

Run the source-backed local comparison in the locked environment.

```bash
python -m pytest tests/unit/test_musical_score_source_tests.py \
    tests/unit/test_musical_score_notebook.py \
    tests/unit/test_drawer_multi_and_score.py -q
```

Run the independent native notebooks with notebook dependencies installed.

```bash
python -m pytest tests/integration/test_register_graph_notebook.py -q
```

External evidence uses prefix `g2-score-source-`, including original tests,
source-definition disposition, focused/regression checks and frozen blob hashes.

## Local verification

Ten new cases pass. The combined Drawer/graph/notebook regression passes 225
cases in 12.89 seconds with eight existing parser deprecation warnings. The
independent native kernels and CPU/tensor-network regression pass 86 cases in
13.76 seconds with two existing warnings. Both unchanged original tests pass
in 2.33 seconds, including the real source notebook kernel. No tests skip.
Strict typing of the added test file, Black, Ruff, Compiler team scope,
architecture, multi-level IR and protected public API checks pass. All three
frozen blobs match the fixed inventory. Production owners are unchanged, so this
batch makes no new production coverage claim.
