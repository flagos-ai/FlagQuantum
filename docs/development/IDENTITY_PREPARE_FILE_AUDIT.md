# Identity signal preparation source family

The unchanged reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The original files retain
Google LLC 2024 Apache-2.0 attribution.

| Inventory | Frozen Git blob | Disposition |
|---|---|---|
| Q-0660 `reflections/__init__.py` | `15780de77b1637deab59d60722c84f30b3d8e75d` | Complete empty initializer audited. There are no declarations, imports or runtime exports to reproduce. |
| Q-0661 `prepare_identity.py` | `cbd8eb35d88d5c02b2428e988e44de6615b7db24` | Complete declaration, factory, direct Identity dependencies and concrete execution mapped to native owners. |
| Q-0662 `prepare_identity_test.py` | `ee6cac40ed0eafc51f69594d94c93172ef70e8f8` | Both original functions and the original factory responsibility covered. |

## Native ownership and interfaces

`algorithms/primitives/identity_prepare.py` owns the `IdentityPrepare`
specialization of the existing `PrepareOracle`. Core continues to own register
metadata and signatures. `PackedPrepareDeclaration` and `BlackBoxPrepare` retain
packed binding, validation and the ordinary append consumer. Compiler's existing
`register_identity_ir` owns physical Identity expansion. No external object enters
these owners. The previous private empty-preparation callback remains available
to its existing consumers with its original optimized-empty behavior.

| Complete source member | Native responsibility |
|---|---|
| `selection_regs` converter | One Register becomes a singleton tuple. Sequences and iterables retain their ordered elements. |
| `from_bitsizes` | Preserve `reg0_`, `reg1_`, and subsequent names, opaque widths and symbolic expressions. |
| `selection_registers` | Return the original ordered descriptors without dtype or direction coercion. |
| `junk_registers`, inherited signature | Reuse `PrepareOracle` defaults and shared signature validation. |
| `build_composite_bloq` | Bind each named concrete scalar quantum port to the existing register Identity emitter, in source signature order. |
| `adjoint` | Return the same instance. |
| `_prepare_identity`, DocSpec | Map the complete 10/4/1 factory through the native declaration and existing binding. |
| inherited coefficient norm | Preserve the missing-normalization exception and bound normalization `None`. |

Logical words use their supplied MSB-first wire order. Noncontiguous/reversed
bindings preserve an entangled spectator. Preparation borrows no work or classical
storage, introduces no measurements, and leaves every selected wire unchanged.
There is one unitary branch. Both declared directions, the complete complex
operator, full complex state and owned IR JSON round trips are verified.

```python
import flagquantum as fq
from flagquantum.algorithms.primitives.identity_prepare import IdentityPrepare

op = IdentityPrepare.from_bitsizes((2, 1)).bind()
circuit = fq.Circuit(4).h(0)
circuit.append(op.make_on(selection=(3, 1, 2), junk=()))
# The spectator superposition and all selected wires are preserved.
```

## Source failures and executable admission

An empty selection collection decomposes successfully. The source call graph
retains that empty root as a terminal, while its direct-call dictionary is empty.
Both the independent zero QEC query and zero peak query succeed.

An explicitly present zero-width register fails source body construction with
`BloqError`, because Identity's signature omits its q port. A shaped selection
fails with `AttributeError` on the ndarray's missing `reg`. LEFT and RIGHT
selections fail finalization with `BloqError`. A classical CBit selection fails
Identity's dtype consistency check with `BloqError`. These unchanged failures
are reference assertions. Native metadata preserves those declarations, while
`bind()` refuses these bodies before emitting a quantum instruction. Native
error messages identify the unsupported binding domain. Symbolic metadata is
retained and concrete execution delegates its width refusal to the existing
packed binder. These failures do not imply successful physical admission.

## Executed primitives and declared costs

For widths 4/4/5 the source direct graph contains Identity(4) twice and Identity(5)
once. The native declaration gives the same keys and multiplicities through the
existing identity cost key. All Identity QEC costs are zero, and the source peak
is thirteen logical qubits. This resource fixture is checked without allocating
an 8192-by-8192 dense operator.

The native explicit physical binding emits thirteen one-wire `i` instructions
for this fixture. The original Cirq conversion retains register-sized Identity
operations, and the existing snapshot importer omits multiwire Identity no-ops.
For widths 2/1 its snapshot therefore contains one `i` on the one-bit register,
while the native explicit binding contains three `i` instructions. The tests
preserve that exact captured stream independently and compare the common source
Identity calls and complete operators. No equality of these differently expanded
primitive inventories is claimed. Width one matches the snapshot instruction
exactly. Source QEC zero cost is never inferred from the emitted instruction count.

The original tests pass five modes and retain one counts-mode skip, whose source
reason is that this oracle only has counts from decomposition. The original
4/4/5 direct-count assertion is independently reproduced above. The concentrated
native regression passes all 77 cases without skips or warnings. Strict typing
of both changed Python files, Black, Ruff, Algorithms scope, architecture,
multilevel IR and protected API checks pass. All three frozen blobs are checked
without changing reference code or original assertions.

Reflection composition belongs to Q-0663/Q-0664 and remains independently open.
