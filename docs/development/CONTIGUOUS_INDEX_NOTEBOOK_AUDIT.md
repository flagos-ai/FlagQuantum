# Complete triangular indexing notebook migration

The original ten-cell notebook at Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389` is restored without edits from
Git blob `104352322042258aaed807779f33bf8611053c77`.
`examples/notebooks/triangular_register_index.ipynb` preserves every original
cell type and topic in order. Two additional cells demonstrate the accepted
native append execution and separate resource accounting. Original copyright
and Apache-2.0 provenance remain in the first cell.

## Complete teaching scope and direct dependencies

| Original topic | Native owner and evidence |
| --- | --- |
| Zero-indexed triangular formula | The original nu*(nu+1)/2+mu expression, zero-based convention and THC citation remain in teaching prose. |
| Four-bit mu/nu and eight-bit s instance | Native unsigned through-port metadata retains the actual source signature, including its occupied result word. |
| Graphical signature | Existing signature-table and Graphviz owners display the original port order and widths. |
| One-level call graph and terminal counts | The existing contiguous-index resource model and generic graph show nineteen Toffoli calls independently of gate emission. |
| Native coherent execution | The existing one-bit append example verifies occupied-target XOR, all complex amplitudes, spectator preservation, clean work and repeated-XOR restoration. Complete multiwidth stream/branch/classical evidence is reused from CONTIGUOUS_INDEX_FILE_AUDIT.md. |
| Original-width emitted inventory | The native columns emitter produces and counts the complete four/eight-bit circuit without allocating a dense state. Its explicit workspace stays caller-owned. |

The acceptance test executes the untouched original notebook, verifies the exact
blob and all original cell topics, then executes the native notebook with
Qualtran/Cirq imports prohibited. Signature order, widths, sides, encodings,
nineteen model Toffoli calls, model root width and emitted inventory are checked.
No production registry or protected API is introduced.

The source root has no gate decomposition. The native XOR execution therefore
retains its own evidence. The source glyph's minus polynomial and classical
callback's missing through-s parameter remain under Q-0190/Q-0191. These source
defects and generic static/symbolic admission boundaries remain recorded in the
complete source audit.

The one-bit native example emits four T, one measurement and one clean work
qubit with six physical wires including a spectator. The source one-bit model
reports one Toffoli, zero measurements and four logical wires. Original-width
model and primitive accounts are displayed separately in the notebook.

Stored notebook outputs are cleared. Rendered signature/graph figures and logs
are stored under `ftqc-reference-env/g2-index-notebook-*`.

At the original four/eight-bit dimensions the columns emitter has 76 T/TDG,
226 CX, nineteen measurements/resets and twenty-two explicit work qubits.
The independent root model has nineteen Toffoli and sixteen logical ports.
The concentrated regression passed 140 cases without skips. Twenty-five
warnings are twenty-four existing paired NumPy overflow warnings and one
IPython fallback. Strict typing checks the acceptance test and all extracted
notebook code. Formatting, lint, Compiler scope, architecture, multilevel IR
and protected API checks passed. Both rendered figures were visually inspected.
