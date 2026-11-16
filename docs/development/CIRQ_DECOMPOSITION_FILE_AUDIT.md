# Cirq import and decomposition source audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The import and decomposition
source and test blobs are pinned in
`tests/unit/test_cirq_decomposition_file.py`. The original paired test suites
passed eleven cases with three source warnings.

| Source responsibility | Native ownership |
| --- | --- |
| One Cirq protocol expansion | The optional Ecosystem boundary creates a fresh greedy manager for each invocation. It uses the gate-specific protocol when given a gate and the value protocol otherwise. It flattens the returned op tree in order and retains the distinction between an empty tree and no decomposition. |
| Carry a concrete decomposition into native IR | The explicit import binds the incoming operand order to native wires. The existing static converter lowers the resulting operations without reordering them. Idle operands retain their native extent. |
| Recognized gate and fallback resource keys | Exact recognized gates remain leaves. An unknown gate with a known decomposition may resolve to native resource leaves. The traversal rejects unresolved leaves, tagged or classically controlled operations and cycles. |
| Work and SDK object boundary | The Cirq manager is local to the protocol invocation. Explicit import rejects newly allocated Qids until they have a native lifetime binding. Only native instructions and string diagnostics cross the adapter. |

The original empty-decomposition test passes for the same gate protocol.
A named and reordered native import compares the entire three-wire unitary,
including the idle wire. Allocator tests compare identity reuse within a
protocol call and isolation between calls. The original symbolic SWAP power
resolves to two CNOTs, two signed Y powers, and the controlled Z power.
Under the shared compiler resource model it declares five Cliffords, one
rotation, one And, and one measurement. These are analytical categories.
The resource query emits no native instructions.

The explicit decomposition route retains the same static gate and measurement
contract as direct Cirq import. A leaf unsupported by that contract still
reports its conversion issue. Resource recognition has a separate purpose
and can recognize a gate even when the static adapter rejects its executable
form.

## Remaining boundaries

Q-0722 and Q-0723 cover the complete one-step source protocol and original
empty-tree test. Q-0717 and Q-0718 retain the original wrapped left-only
and asymmetric gates, tensor fallback, arbitrary global phase and allocated
op-tree import. The original wrapper keeps SDK gate objects. The local
adapter confines those objects to the optional Ecosystem layer. Import of
ordinary Cirq circuits retains the existing canonical-wire API and contract.
