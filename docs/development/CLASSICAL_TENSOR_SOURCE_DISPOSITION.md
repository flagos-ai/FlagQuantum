# Classical action tensor source disposition

Reference: unchanged Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, complete
`qualtran/simulation/tensor/_tensor_from_classical.py`. There are four top-level
functions and one nested index iterator. Existing native owners retain
Apache-2.0 attribution. This is a bounded single-device numerical inspection
path, not an automatic quantum decomposition.

## Complete ownership and interface mapping

| Source definition | Native owner and disposition |
| --- | --- |
| `_bits_to_classical_reg_data` | `classical_register_codec.decode_register_bits` composes existing dtype-specific bit/word/field codecs. Enumeration supplies flat scalar bits or `shape + (num_qubits,)` array bits. Explicit caller codecs retain precedence. |
| `_bloq_to_dense_via_classical_action` | `_classical_action_tensor_data` enumerates inputs in signature/row-major/MSB-first order, calls the supplied action and fills a zero/one tensor with output axes before input axes. It retains output truthiness, zip truncation and NumPy indexing behavior. |
| `bloq_to_dense_via_classical_action` | `classical_action_tensor` adds operation context and cause only to ValueError, then groups to scalar/vector/matrix. Finite native metadata validation remains an earlier boundary. |
| `my_tensors_from_classical_action` and nested `_signature_to_inds` | `classical_action_tensor_node` reuses the raw enumerator and existing register-axis binder. It attaches quantum output/input labels and preserves raw callback failures. Classical ports are explicitly refused before callbacks rather than constructing the source's incompatible data/index ranks. |
| Imports and type-only foreign references | Existing NumPy, Torch, Core signature, codec and callback owners replace foreign Bloq/Connection/Quimb types. No package initializer, global provider registry, public API or new contract is added. |

The action and codecs are caller-owned numerical callbacks. Supplier selection
continues to use the existing explicit provider table; no function name is used
to infer phase or a classical action. Tensor data follows the caller's dtype and
device after binding. The NumPy action itself is not trainable; gradients remain
available through subsequent native contraction with trainable inputs.

## Direction, classical axes and empty arrays

Thirty source-backed cases cover quantum/classical bits, LEFT/RIGHT/THRU sides,
and scalar, `(2,)`, `(1,2)`, `(0,)`, `(2,0)` shapes. They compare the unchanged
source's exact dense outputs or failure type/message/cause class and the complete
callback prefix, then compare raw quantum leaf data/indices or raw failures.
The raw action fixture intentionally isolates enumeration from foreign builder
validation; it is not proof of arbitrary external Bloq support.

For scalar classical bits, dense enumeration can succeed. Array inputs still
reshape using **quantum** width, while the tensor allocation counts **all** bits.
Classical arrays therefore fail before the first action when those dimensions
disagree. Classical output arrays may produce a dense table but the source leaf
still supplies only quantum index names, leading to incompatible ranks. Empty
arrays reach the existing source codec's empty-vectorization failures, including
zero-width edge cases; the reference is not patched or silently repaired.
Native leaf refusal remains earlier than those raw foreign failures. Tests
verify that native refusal executes no action callback.

The original six `_tensor_from_classical_test.py` cases already have unchanged
execution and native table/node replacements in `test_tensor_protocol_source_audit`.
Existing tests separately cover all original outputs, tuple truncation and bad
indices, exact raw exception identity, dense ValueError cause, non-injective
actions, over-40-bit refusal before enumeration, and signed/typed codecs.

## Phase and execution boundaries

A classical value mapping cannot recover a quantum phase. Source Z's classical
identity becomes an identity table, which differs from its quantum Z matrix;
Hadamard has no applicable deterministic classical action. Existing positive
and negative source tests retain those facts. Native enumeration neither
normalizes a non-injective table nor claims it is unitary. Known phase-free
X/swap/Toffoli/Add/Xor tables are compared with the unchanged source quantum
path, complete complex input amplitudes and phase gradients.

Existing measured-Add execution checks preserve actual gate order, operands,
feedback, all small measurement branches, clean work and `circuit.append(op)`.
Those checks are rerun as dependencies; this audit introduces no new physical
algorithm. Q-0895 remains open under closure rule5 because the classical-index,
empty-codec and phase-information boundaries remain explicit. Original-test
parent Q-0896 stays accepted; a new test count does not close the implementation.

## Resource accounting

Actual primitives, measurements, feedback slots and work bits **added by this
audit: zero**. The original 40-total-bit ceiling is retained; dense float64
storage is `8 * 2**(left_bits + right_bits)` bytes. It is a refusal ceiling,
not a safe memory-allocation promise. The new layouts use at most four axes
and 16 float64 values (128 bytes), excluding Python/process overhead.

Quantum gate model for enumeration: **not applicable**. Existing measured-Add
actual primitives and source QEC estimates remain separate in their original
evidence; a dense table entry is not an emitted gate. No GPU, capacity or full
repository certification is claimed; the historical default-suite timeout is
not rerun unchanged.

Next source file: Q-0882 `classical_sim.py`, complete state/truth-table and
foreign error disposition using the existing component classical owners.
Q-0893 already has a complete six-function source audit; reuse that evidence
rather than repeating projection or enumeration. Do not introduce foreign dtypes.

Local acceptance: **30 new scenarios**, **2,692 related tests in109.63 s**,
**84 CPU/TN tests in11.46 s**, and **six unchanged original tests** pass.
Strict mypy, Black/Ruff, Simulation scope, architecture, IR, protected API and
Bell example pass. The enumerator owner has **52 statements/10 branches at100%**.
Three frozen source blobs match the inventory. Acceptance `cd1648e1`. Existing
source warnings and default timeout remain; this is not full-suite certification.
