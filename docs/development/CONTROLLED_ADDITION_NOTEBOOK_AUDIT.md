# Complete controlled addition notebook migration

The original eleven-cell notebook at Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389` is restored without edits from
Git blob `d4235893521075c7a25d1a00f586dd39e2cd86f2`.
`examples/notebooks/register_controlled_addition.ipynb` preserves every original
cell type and topic in order. Two additional cells demonstrate native append
execution and separate resource accounting. Original copyright and Apache-2.0
provenance are retained.

## Complete topic coverage and direct dependencies

| Original topic | Native owner and evidence |
| --- | --- |
| Controlled n-bit addition and 2n-1 Toffoli explanation | Existing CAdd decomposition and resource owners remain authoritative. The original algorithm and activation parameters are preserved in teaching prose. |
| Three-bit small instance | ArithmeticCall and typed register metadata retain ctrl/a/b order, activation one and unsigned three-bit words. |
| Thousand-bit large instance | The same metadata retains the complete width without allocating a dense state. Its source Toffoli declaration and source peak-query failure are recorded in CONTROLLED_ADDITION_FILE_AUDIT.md. |
| Graphical signatures for both instances | The existing signature-table renderer preserves bit/word encoding and widths. |
| Small one-level graph | Existing arithmetic declarations and generic graph owners retain three And, one Add and three And-adjoint calls. The exact original direct child identities are compared with the native graph. |
| Coherent append and exact inverse | Existing DynamicCircuit, FieldOperation, controlled_add_ir and controlled_add_adjoint_ir exercise every three-bit input and a coherent spectator. Five clean work qubits are restored after each direction. |

The original notebook executes completely through its signature-only drawing
route. This is distinct from the source full musical-score API mismatch
recorded in the complete source audit. The acceptance test executes the original
asset unmodified, then executes the native notebook with Qualtran/Cirq imports
prohibited. It checks both complete instance signatures and the original direct
graph. Native execution verifies complete complex amplitudes in measured
trajectories and exact inverse restoration. Existing complete source capture
and exhaustive raw-branch evidence is reused from the file audit.

The three-bit stream emits twenty T/TDG, five measurements/resets and five
explicit clean work qubits. The independent source model has peak twelve.
The spectator raises the notebook execution's physical width to thirteen.
Primitive counts and model fields are displayed separately. Symbolic physical
admission, static converters, generic outer control, source narrow-signed
semantics and large-width source recursion remain under Q-0186/Q-0187.

The notebook introduces no production registry or protected API. Stored outputs
are cleared. Rendered signature and graph figures are inspected separately.
Logs are `ftqc-reference-env/g2-controlled-addition-notebook-*.log`.

Concentrated regression passed 656 cases without skips. The single warning is
the existing IPython directory fallback. Strict typing checks the acceptance
test and every extracted code cell. Formatting, lint, Compiler scope,
architecture, multilevel IR and protected API checks passed. Both rendered
signature/graph figures were visually inspected.
