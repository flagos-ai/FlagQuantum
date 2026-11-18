# Complete conditional arithmetic notebook migration

The locked Qualtran v0.7.0 notebook at commit
`bcb17bb25360ed68ce3bc084645db3e929799389` is restored unmodified from Git blob
`97da1cd23d35b8535e5da1b9a4e644ffcd16c869`. All twelve original cell types,
example order and teaching topics are retained in
`examples/notebooks/register_conditional_arithmetic.ipynb`. Two additional
cells demonstrate the existing native append execution and separate resources.
The copyright and Apache-2.0 provenance remain in the notebook.

## Complete teaching scope and dependencies

| Source topic | Native owner and evidence |
| --- | --- |
| Control-sector transformation and complement/Add/complement algorithm | The original explanation is retained. False polarity prose is corrected to add on zero and subtract on one, following the actual source body and adjoint. |
| Unsigned eight-bit and signed eight-bit examples | Native register signatures retain ctrl/a/b order, widths and encoding. Control is represented as a one-bit unsigned register. |
| Signed symbolic n-bit example | Native metadata and the existing symbolic resource model retain the declaration without claiming physical symbolic admission. |
| Three graphical signatures | Existing register table and Graphviz owners show all widths and encodings in the original display order. |
| Symbolic one-level call graph | The original two X calls, two controlled BitwiseNot calls and one Add are preserved as typed native child identities. The notebook-local example record supplies the root metadata, with no production registry. |
| Native coherent execution and inverse | The existing conditional arithmetic example uses circuit.append(op), measured trajectory execution and opposite-polarity inverse restoration. Complete branch/source tests are reused from CONTROLLED_ADD_OR_SUBTRACT_FILE_AUDIT.md. |
| Resource accounts | Eight-bit measured inventory contains 28 T/TDG, 97 CX, two X, seven measurements/resets and seven clean carries. Independent source formula reports its categories and peak 24 separately. |

The acceptance test executes the entire untouched source notebook, compares all
three original instances and direct child multiplicities, then executes the
native notebook with Qualtran/Cirq imports prohibited. It checks five display
objects, all register widths and encodings, primitive inventory and model peak.
The helper record and rendering functions stay in the teaching asset. Compiler
emission, resource formulas, Dynamic Runtime execution and Drawer remain under
their accepted owners.

Q-0183/Q-0184 retain the source narrow-signed classical/circuit inconsistency,
symbolic Program admission and static converter boundaries. Their original
unimplemented count fixture also remains explicit. Native whole-state and raw
measurement-branch evidence is reused, including both polarities and exact
opposite-polarity adjoints. No reference code or source blob is edited.

The rendered signature and graph figures are inspected separately. Stored
notebook outputs are cleared. Logs are
`ftqc-reference-env/g2-conditional-notebook-*.log`.

The concentrated regression passed 258 cases without skips. The single warning
is the existing IPython unwritable-directory fallback. Strict typing covers the
acceptance test and every extracted notebook code cell. Formatting, lint,
Compiler scope, architecture, multilevel IR and protected API checks passed.
Both signature and call-graph figures were visually inspected.
