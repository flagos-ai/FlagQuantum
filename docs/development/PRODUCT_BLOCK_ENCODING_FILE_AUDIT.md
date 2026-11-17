# Product block encoding source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen files under `qualtran/bloqs/block_encoding/` are Apache-2.0,
copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0291 | `product.ipynb` | `727fb156b779618d838b430868681e886a902ca6` | Closed by the complete native tutorial. |
| Q-0292 | `product.py` | `0c806922e9d734f7a2d823c960369c1a443ebb4a` | Symbolic signature and fallback costs verified. External-source objects and symbolic execution remain open, with the locked unequal resource failure recorded. |
| Q-0293 | `product_test.py` | `389ffb997e324e16e3bb64e87e9bbd274188bdbe` | Closed by the original scenarios and existing measured flag evidence. |

## Port and execution rules

System ports are shared. The ancilla consists of the leading success flags
followed by the maximum child signal width. The resource register is shared
under the source zero-restoration contract. Child and flag temporaries use
their assigned clean work and slots while active; they can reuse sequential
allocations. The children execute right to left in the forward direction.
The adjoint reverses that order and uses each bound child adjoint. The
product's logical flags and signal register carry failure information.

| Original scenario | Native evidence |
| --- | --- |
| Default, auxiliary-width and symbolic instances | Original ordered concrete widths and scalar metadata, original symbolic ancilla width `1 + Max(a1, a2)`, normalization `alpha1*alpha2` and weighted error. The example states the original symbolic decomposition refusal. |
| Constructor and signal checks | Existing native empty, mismatched system and nonidentity-signal refusals, plus the original identity preparation through each concrete fixture. |
| Default, single and auxiliary tensor tests | Original noncommuting T and H and the existing single-child T test. Both original two-child fixtures match the complete forward and adjoint matrices and round-trip the emitted native IR. |
| Nested Cirq test | The native nested Product emits three X gates on the original system wire. Its full three-wire matrix matches the original Cirq export with explicit qubit order. |
| Seeded dense tests | All ten original seed-1234 gate families retain the original width, count and child tensor, with full native complex matrices compared in both directions. |
| Counts | Original direct and decomposition routes remain independently declared, including different flag polarities. Existing tests compare both source QEC profiles and logical widths. |
| Notebook | Unchanged 12-cell source and 14-cell native tutorial independently execute, with all three example signatures, the direct graph, and the complete native append and inverse. |

The ordinary source tutorial fixture emits H followed by T with one idle
success flag and no work or measurements. Its source model charges one
Clifford, one T and two peak logical qubits. The zero-ancilla child omits
the flag check. Prior acceptance covers nontrivial leaking signal blocks,
measured children and all enumerated temporary-work histories. See
[Product block encoding](PRODUCT_BLOCK_ENCODING.md) for the measured example
and the direct versus decomposed flag model.

The original non-notebook baseline passes 11 cases, with two mode skips and
two expected failures for the incomplete source counts and serialization
modes. The verified codec is the native emitted IR. The original unequal
nonzero resource partition failure is retained as an explicit source
execution and resource-query boundary.

## Symbolic resource admission

The packed declaration retains the common system, one flag per child
except the last, maximum child ancilla and maximum resource width.
The symbolic resource query retains the locked fallback over each
callee peak and the outer signature. A callee's ancilla padding is
charged only when the child exposes an ancilla port. Symbolic
MultiControlX uses the shared predicate model and its signature peak.
The concrete Product path remains covered by the tensor and primitive
checks above.

For the original two-child example with ancilla widths a1 and a2,
the native expression matches the source signature
`1 + Max(a1, a2)` for the packed ancilla. The modern source QEC
categories are one T, `a2 - 1` And and measurement, and
`a2 + 2` Cliffords. The peak is compared as a separate symbolic
expression. The query also covers zero or fixed child widths, shared
symbolic resources and symbolic system width under both profiles.
The Unitary child formula retains zero and partition width
intermediates when computing its own symbolic peak.

A child with a nonzero resource declaration exposes that width at the
source partition boundary. When it also requires a tail to reach the
shared resource width, its declared partition is inconsistent.
The original QEC query rejects this for unequal concrete resources
and for a symbolic resource whose equality to the shared width is
unproven. The native declaration remains available. Numerical binding
and the affected resource query retain the explicit refusal.
These are source admission boundaries. Foreign source object binding
and serialization remain open.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_product_block_encoding.ipynb
```
