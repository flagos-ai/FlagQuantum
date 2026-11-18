# Complete k-XOR instance-loading notebook

Q-0616 maps all eleven original cells of Qualtran v0.7.0
`optimization/k_xor_sat/load_kxor_instance.ipynb`, frozen blob
`9a08818e673ccc5621e4a077ca1fab1f840a4075`, to the native notebook
`examples/notebooks/kxor_instance_loading.ipynb`. One additional evidence cell
binds the complete original concrete example through `circuit.append(op)`.
Each original cell index remains in notebook metadata.

The symbolic and concrete factories, exact example clauses, scope ordering,
register sides and widths, graphical signatures and one-level QROM call graph
match the untouched original notebook. Both signatures and the call graph render
with existing native drawing owners. Every code cell executes independently
with Qualtran, Cirq and OpenFermion imports blocked in the native execution.
The unmodified original notebook also executes independently.

The symbolic model retains a single packed target of k variable-index words.
Its And, Clifford and measurement expressions match every original category.
The concrete evidence emits 86 primitives and three measurements with two clean
work wires and two reusable classical slots. Its QEC model records three Ands
and three measurements independently of primitive lowering.

Complete source stream, amplitude and cleanup evidence is reused from
`KXOR_LOADING_FILE_AUDIT.md`. That audit checks the exact original example stream
and a smaller coherent fixture over every forward and joint measurement history.
The notebook carries the existing index/RHS declaration-only execution boundary.
Q-0617 remains open for root control-adder wiring and Q-0618 retains the original
count/serialization failures. Closing notebook adaptation preserves these gaps.

## Validation

The concentrated regression passed 482 tests with three retained original warnings.
Strict mypy passed for the complete extracted notebook and its test. Black, Ruff,
Algorithms scope, architecture, multi-level IR and the protected API snapshot
passed. The symbolic signature, concrete signature and one-level graph were
rendered to PNG and visually inspected together.
