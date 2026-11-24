# Tensor-product block encoding source files

The shared typed graph, explicit inverse ownership and original tuple-of-Bloq
wire failures are documented in
[the packed-block constructor audit](RESOLVER_PACKED_BLOCK_AUDIT.md).
Direct native graph parity does not imply successful original serialization.

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen files under `qualtran/bloqs/block_encoding/` are Apache-2.0,
copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0302 | `tensor_product.ipynb` | `f3f3652c6dd29f21f8151392c164267d689ae7b5` | Closed by the complete native tutorial. |
| Q-0303 | `tensor_product.py` | `d1a01a8eccee2fff1f319668b6bd34f4a3890d13` | Complete native composition and external shaped binding verified, with unresolved source and zero-system failures explicit. |
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
skips and two expected serialization failures.

## Symbolic admission and cost

The native declaration packs the sum of each category of child ports and
omits zero-width ports. The source rejects symbolic numerical decomposition.
Its QEC fallback sums child gate categories and takes the maximum of the
outer signature width and the individual logical peaks. The concrete peak
accounts for bystander widths separately. Tests compare both QEC profiles
and symbolic peaks for unknown system, ancilla and resource widths.

The source inherits the register-name label from its base object; the existing
native register graph and tutorial retain those labels without adding a
TensorProduct-specific renderer. A source tensor mixing a zero-width identity
with a nonempty system raises a missing system-port error during partition
construction. A completely zero-system source fails before its body because
the required `system` argument is absent. Native numerical import explicitly
rejects a zero-system block rather than changing Circuit admission.

## Whole-source external binding

`TensorProductBlockEncoding.of(*children)` retains the ordered variadic factory.
The tuple constructor owns nonempty validation; signature, five scalar
properties and signal preparation retain source declarations. `make_on`
replaces source split/join partition bookkeeping with the shared native packed
layout owner, retains child order, and captures independently bound children.
The model calls retain grouping and charge bystander ports independently of
emitted primitives. Existing generic register rendering covers the inherited
wire labels, and both source and native tutorials remain teaching evidence.

The optional Ecosystem snapshot boundary now accepts concrete shaped ports.
It checks equal input/output names, total widths and exact shapes before
export, constructs source qubit arrays in row-major order, validates returned
arrays against that same shape, then flattens them to native through ports.
The original source arrays and SDK objects do not escape the adapter. A
symbolic dimension or mismatched sided shape is rejected; equal flattened
width alone does not establish matching ports. Return-wire restoration and
clean allocation ownership remain the existing shared implementations.

`test_tensor_product_external_file.py` captures both actual original concrete
root factories in all four directions, with independent locked resource
declarations, complete complex matrices and native JSON roundtrips. Both
resolved symbolic width pairs replace native consumer children with actual
external T/H snapshots in both directions. Metadata, substituted signatures
and independent costs match; unresolved original decomposition still fails.

An independent original BlockEncoding with multidimensional system, ancilla
and resource ports exercises the source's real TensorProduct Partition route.
Direct and composed root captures match all four complete matrices, including
occupied logical auxiliaries; no matrix synthesis fallback is used. A measured
Add/T root checks each complete branch operator on every logical basis input
and a complex coherent state, with private work zero, including JSON roundtrip.
This tests the actual moment-scheduled source stream unchanged. The canonical
contiguous-macro certificate does not parse its interleaved disjoint gates;
its failed trial is not counted as passing. The two numerical histories here
are exhaustive, and earlier canonical native composition evidence is reused.

Actual measured Add/T emission has five T/T-dagger instructions and one
measurement. Its independent logical model declares one And and one T;
the four-T lowering accounts for the difference. Snapshot reserved wires,
returned-wire swaps and source logical peak remain separate statistics.
All three original protobuf factory checks fail on object arrays with
`allow_pickle=False`; the reference is unchanged. Native IR JSON is a distinct
supported serialization contract.

The batch regression passes 308 Product/TensorProduct, block snapshot,
external sparse oracle and feedback tests with no skips. The final shaped
native-consumer addition passes the ten whole-file tests. Strict mypy passes
all three touched production files; Black, Ruff, both owning team scopes,
architecture, multilevel IR and public API checks pass. The unchanged native
TensorProduct tutorial executes in its independent kernel with eight code
cells and two SVG outputs. The three frozen upstream asset blobs are checked
by the whole-file fixture; earlier original test/notebook evidence is reused.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_tensor_product_block_encoding.ipynb
```
