# Tensor-product block encoding source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen files under `qualtran/bloqs/block_encoding/` are Apache-2.0,
copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0302 | `tensor_product.ipynb` | `f3f3652c6dd29f21f8151392c164267d689ae7b5` | Closed by the complete native tutorial. |
| Q-0303 | `tensor_product.py` | `d1a01a8eccee2fff1f319668b6bd34f4a3890d13` | Open for generic symbolic port admission and external child-object protocols. |
| Q-0304 | `tensor_product_test.py` | `1f997603c83beb1ca274a42e5c7ce102335c41e2` | Closed for the original runnable scenarios. The original fixture-mode outcomes are recorded below. |

## Ownership and reused evidence

The Algorithms TensorProduct owner composes existing block children through
the Compiler's packed layout. System, ancilla and resource words each keep
child tuple order and big-endian bit order within a child. Forward append
preserves child calls. Adjoint append reverses order and requests each child
adjoint. The caller owns work and result slots, and each child owns cleanup.
Ports passed through to a unitary child remain untouched. The source scale
multiplies child scales, and its error adds the immediate child products of
scale and error. Retain nested grouping for the latter.

The prior [execution audit](TENSOR_PRODUCT_BLOCK_ENCODING.md) verifies 64
ordered child streams, 128 resource comparisons across both QEC profiles,
noncontiguous bindings, complete complex tensors, all small measurement
histories, and work-wire cleanup. The existing model charges other children
as bystanders and takes the peak of sequential work. In the existing Add/T
example, five emitted T/T-dagger instructions and one measurement match
five model T equivalents and one model measurement. Six reserved wires and
the six-qubit logical peak are distinct quantities.

## Complete original tutorial and scenario map

| Source scenario | Native evidence |
| --- | --- |
| Three fixture instances and signatures | Concrete T/H, overridden T/CNOT with packed widths 3/3/2, and symbolic T/H with the declared `a1 + a2` ancilla width. An empty tensor product is rejected. |
| Parameters and graphical call declarations | Concrete and symbolic normalization/error, two ordered unitary children, model-backed call graph and three signatures. |
| Single T and T/H tensors | Prior single-child complex cases and the tutorial's full T/H matrix and adjoint round trip. |
| T/CNOT override projection | Exact T/CNOT operands and full 256-by-256 complex matrix against the locked tensor. Tensoring the projected system matrix with identity verifies all 32 auxiliary configurations. |
| Nested X/X and CNOT export | Existing Cirq boundary emits identical X, X and CNOT operands. Full exported matrices and the original 0111 to 1010 basis case agree. |
| Signal preparation | Reuse the existing identity-only contract and source-matched refusal of nonidentity child preparations. |
| Notebook execution | Unchanged 12-cell original and 14-cell native tutorial executed in independent kernels. Native kernel rejects reference imports and checks both SVG outputs. |

The unchanged original non-notebook run has seven passes, two fixture-mode
skips and two expected serialization failures. The symbolic ancilla is
shown as a register declaration in the tutorial. Executable child binding
still requires concrete widths. There is no generic admission for arbitrary
external child objects, source wire-symbol hooks, or a fully zero-wire
numerical product.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_tensor_product_block_encoding.ipynb
```
