# Cirq-to-native conversion source-family audit

This work follows Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
Q-0717 and Q-0718 track `cirq_interop/_cirq_to_bloq.py` and its original tests.

| Source responsibility | Native evidence and remaining boundary |
| --- | --- |
| Gate recognition and resource estimates | `resource_keys.py` covers exact gates, controlled binary clauses, parameterized powers and the admitted inverse leaves. Ecosystem owns opaque SDK values. |
| One-step decomposition and qubit manager | `_decompose.py` and `_qubit_manager.py` preserve protocol order and distinguish no decomposition from an empty decomposition. Native register import uses declared clean lifetimes for admitted gate operands. |
| `cirq_optree_to_cbloq` | `import_cirq_register_optree` handles default through ports and explicit concrete signatures with shaped bit-axis quregs. It reuses the native builder for split/join, first-use allocations, clean releases and sided output binding. |
| Gate wrappers, circuit conversion and `decompose_from_cirq_style_method` | Tagged and unsupported gates are refused. Automatic port extraction from arbitrary external wrappers, gate-supplied internal lifetimes and wrapper tensor/effect dispatch remain open. |
| `_QReg` and register helpers | Native port bit order, shape and ownership are checked. External gate-declared multi-bit grouping and effect consumption remain open. |
| `_my_tensors_from_gate` and wire symbols | Ordinary admitted gates use native execution. General wrapper tensor/effect and diagram protocol import remain open. |
| Global phase and unsupported dimensions | The typed import lowers concrete global phase through the existing scalar primitive on a live wire. Anchorless or symbolic phases, wider qudits and symbolic signatures require separate admission. |

All ten frozen original tests execute unchanged. The native fixture compares
shaped ports, two-sided remapping, a right-only output, allocation and release
against the source optree contract. Physical binding applies one H, three
CNOTs and one T in the same dependency order and preserves all complex
amplitudes after append. The owned work wire returns to zero. The separate
locked model reports four Clifford and one T. Independent gates can be
scheduled by the native binder.

A separate fixture preserves the source's concrete complex phase without an
allocation. Q-0717 and Q-0718 remain open for the wrapper, anchorless phase,
tensor and per-gate lifetime cases above. The plain physical Cirq importer keeps its existing
entry point. The new optree adapter is internal to Ecosystem.
