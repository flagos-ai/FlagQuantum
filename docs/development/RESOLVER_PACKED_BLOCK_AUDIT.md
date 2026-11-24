# Packed unitary and tensor-product block composition

Reference: unchanged Qualtran v0.7.0,
`bcb17bb25360ed68ce3bc084645db3e929799389`, Apache-2.0, Google LLC.

| Source | Frozen Git blob |
| --- | --- |
| `block_encoding/unitary.py` | `bbcff63d78cbd58979bee3670597a778fdd4bdd2` |
| `block_encoding/unitary_test.py` | `4bead193a26eb24303fb96a15bd2233774924e0e` |
| `block_encoding/tensor_product.py` | `d1a01a8eccee2fff1f319668b6bd34f4a3890d13` |
| `block_encoding/tensor_product_test.py` | `1f997603c83beb1ca274a42e5c7ce102335c41e2` |

## Ownership and source order

Compiler layout owns packed system/ancilla/resource signatures and typed block
composition in `packed_block_component.py`. Existing Algorithms common block,
Unitary and TensorProduct signature helpers delegate there. Their metadata,
signal preparation and independent cost models retain their existing owners.
Compiler executes no numerical kernel and selects no backend.

Unitary calls the shared AutoPartition owner exactly once. Only the system
word is unpacked; ancilla and resource remain untouched caller-owned bystanders.
The shared packing owner now retains such extra outer ports. It still requires
explicit exact child inverses, and generic adjoints reverse the original calls.

TensorProduct partitions system, then present ancilla and resource groups,
calls children in original order, and rejoins groups in that same order. Child
subwords use original `system{i}_`, `ancilla{i}_` and `resource{i}_` names.
All exposed words are caller-owned, MSB first. The wrapper allocates no private
work; individual children retain allocation, measurement and cleanup ownership.
Exact inverse construction uses explicit child pairs and the existing directional
Partition owner. Symbolic tensor products retain a typed declaration and refuse
physical execution, matching the original missing symbolic decomposition.

Formats binds original Unitary constructors with a scoped exact-child provider.
Existing fixed-gate factories supply known native forward/inverse pairs; other
children require explicit complete native pairs. No source SDK objects enter
Compiler, and the resolver performs no semantic inference from resource counts.
Ordinary execution still uses `circuit.append(op)`.

## Evidence

`tests/unit/test_packed_block_component.py` compares original recursive calls,
named operands, ordered outputs and complete complex operators. It covers both
Unitary examples, both concrete TensorProduct examples, exact adjoints, coherent
auxiliary bystanders and native child replacement. A non-Hermitian measured
child runs through both wrappers: forward and inverse full operators agree with
the unchanged source, every combined raw measurement history restores the full
complex input and private work is zero. Source imports are blocked during
Unitary resolution, binding and append. Existing Algorithms regression checks
the shared signature-owner replacement without changing consumers.

Receipts are `ftqc-reference-env/g2-unitary-tensor-constructor-*` beside the
worktree. Intake verifies all four blobs and all five original factories in
both directions at three serialization depths. Resources verify Qualtran 0.7.0
and an actual Kickmix 0.1 Simulator. Emitted primitive counts and logical live
widths remain separate from original QEC and qubit models. Default/property
Unitary emits one T (inverse Tdag) on one/four caller wires. Concrete tensor
examples emit T plus H on two wires or T plus CX on eight wires. Wrapping adds
no primitive, measurement or private work.

## Original failures and remaining boundaries

- Original Unitary examples serialize only at depth zero; deeper AutoPartition
  tuples fail the unchanged ragged-array codec.
- All original TensorProduct factories fail tuple-of-Bloq object-array
  serialization, including concrete ones. Direct native graph and numerical
  parity is verified, but no successful source constructor wire is claimed and
  no replacement wire encoding is invented.
- Source Unitary sums element bitsizes without shape multiplicity. Wrapping
  Toffoli declares two system bits although its child needs three. Shallow
  source decomposition succeeds; tensor contraction and further flattening
  fail with a partition-width error. Native typed composition rejects that
  mismatch earlier. The source is neither repaired nor reported as refusing
  shallow decomposition.
- Arbitrary child closure owners, symbolic physical execution, original
  zero-system decomposition and whole-repository/GPU/performance gates remain
  separate. This does not claim generic constructor catalogue closure.

From the repository root in the locked environment, preimport torch and run:

```bash
python -m pytest tests/unit/test_packed_block_component.py -q
```
