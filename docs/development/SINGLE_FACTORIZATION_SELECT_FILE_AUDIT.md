# Single-factorization SELECT audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, under
`qualtran/bloqs/chemistry/sf`.

| Inventory | Source file | Git blob |
| --- | --- | --- |
| Q-0402 | `select_bloq.py` | `030a498a777bb73ca6f9600e8ad00d578e044010` |
| Q-0403 | `select_bloq_test.py` | `0c4548fb131e76283759e302f7d3d23e5abcda86` |

The source declares through registers `p`, `q`, `spin`, `succ_pq`,
and `succ_l`. The address words are absent for two spin orbitals.
Both settings of `additional_control` yield identical signatures and
cost graphs in the frozen implementation. The native source model
now derives the local width from the same order and encoding.
The tests compare both settings at 2, 6, 10, and 16 spin orbitals,
with exact direct Toffoli multiplicity, gate categories, and peak.
The pinned ten-orbital example declares 16 Toffoli and a nine-qubit
peak.

The pinned component supplies no executable decomposition. The
existing native executable instead binds the two or three success
controls needed by the paper operator, and can apply a coherent
coefficient sign. It uses big-endian spatial selection and
orbital-then-spin system order. The matching physical operation attaches
through `circuit.append(op)`.

The native tests compare complete unnormalized complex amplitudes
for all balanced measurement histories at two and four spin orbitals.
They include coherent success flags and one case with both coefficient
sign values. Their expected branch states have zero amplitude on every
nonzero-work basis index. At six spin orbitals, a second append starts
its measurements after the first append's feedback slots.

With four spin orbitals the two-control fixture emits 28 T/T† and
seven measurements with four work wires. The three-control fixture
emits 32 T/T† and eight measurements with five work wires. The model
declares four Toffoli and zero measurements in both cases. Source
budgets and physical counts are retained as separate evidence.
The physical binding assumes populated spatial addresses.

The unchanged original tests return two passes and three skips. The
concentrated regression verifies 47 cases. Strict typing, format,
ownership, architecture, IR, and public API checks pass. This closes
Q-0402 and Q-0403 in the original source scope.
