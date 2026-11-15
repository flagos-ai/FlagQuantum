# Product block encoding source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen files under `qualtran/bloqs/block_encoding/` are Apache-2.0,
copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0291 | `product.ipynb` | `727fb156b779618d838b430868681e886a902ca6` | Closed by the complete native tutorial. |
| Q-0292 | `product.py` | `0c806922e9d734f7a2d823c960369c1a443ebb4a` | Open for generic symbolic binding, external-source object protocols, and unequal nonzero resource partitions. |
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

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_product_block_encoding.ipynb
```
