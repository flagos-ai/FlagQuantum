# Complete register bitwise notebook migration

The original Qualtran v0.7.0 notebook at
`bcb17bb25360ed68ce3bc084645db3e929799389` is restored without edits from Git
blob `1057f0d643ffeb5c7c9f507f5ba1043913c5aa1a`. The original asset has
20 cells. `examples/notebooks/register_bitwise.ipynb` preserves all 20 cell
types and topic order, with two additional cells demonstrating native execution.
Copyright and Apache-2.0 attribution remain in its first cell.

## Ownership and source coverage

| Complete source topic | Native placement and evidence |
| --- | --- |
| XOR declaration, concrete QAny(4), symbolic QAny(n) | Existing ArithmeticCall resource identity and opaque register signatures are kept separately. The notebook displays both signatures with their widths. |
| XOR graphical signature and one-level graph | Existing addition signature renderer and generic call-graph drawer render native metadata. Four CNOT leaves match the original direct edge. |
| NOT declaration, concrete QUInt(4), symbolic QUInt(n) | Existing unsigned signatures and bitwise-not resource identity retain all original examples. |
| NOT graphical signature and one-level graph | One OnEach(4, X) child is displayed at the original depth. Expanding this child yields four primitive X gates. |
| Native append execution | Existing XOR emitter is appended to a Circuit and run on a coherent state with a spectator. Applying it twice restores the full complex state. |
| Resource accounts | Measured four CX for XOR and four X for complement are separate from one-level model edge multiplicities. Both require zero work, measurements and T gates. |

The native notebook uses existing Compiler metadata/emission, Simulation and
Drawer owners. The small display helpers live in the notebook. The optional
IPython boundary is explicitly typed. Runtime execution of the native asset
rejects imports of Qualtran and Cirq. The original notebook executes separately
and unmodified against the frozen reference environment.

The focused acceptance test checks the exact source blob, complete cell order,
all four signature dtype/port combinations, both graph leaf multiplicities,
seven display results and primitive counts. Full bitwise factory streams,
operands, reversed bindings, complex amplitudes, classical edge cases and
symbolic decomposition boundaries reuse BITWISE_FILE_AUDIT.md and its
concentrated regression. The symbolic examples demonstrate declarations and
models. Their concrete Program admission gaps remain under Q-0177/Q-0178.

All four rendered SVG/PNG figures were inspected for readable type/width rows
and compact graph labels. Notebooks are committed with cleared execution
outputs. Batch logs are `ftqc-reference-env/g2-bitwise-notebook-*.log`.

The focused original/native notebook execution passed. Concentrated regression
passed 479 cases without skips. Eleven warnings comprise eight paired source
and native NumPy overflow warnings, two existing IPython directory fallbacks
and one existing noninteractive addition figure warning. Strict typing checks
both the acceptance test and all extracted notebook code. Formatting, lint,
Compiler scope, architecture, multilevel IR and protected API checks passed.
