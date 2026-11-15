# Linear combination source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen files under `qualtran/bloqs/block_encoding/` are Apache-2.0,
copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0285 | `linear_combination.ipynb` | `e44e694d55cc4d9524756741665b68b38cfce860` | Closed by the full native tutorial. |
| Q-0286 | `linear_combination.py` | `a3d8aa7fafa7a6f3fcaf095826751209427cd255` | Open for generic source symbolic/foreign objects and returned-adder boundaries. |
| Q-0287 | `linear_combination_test.py` | `bed60bc7ade281fca1e5b13e341e8b4dd07aabd9` | Open for native runs of the original nine-bit approximation families, random dense children, and remaining exact fixture combinations. |

## Binding and decomposition

The block binds system, coefficient-selection plus constituent-signal
ancilla, and constituent-resource plus coefficient-preparation junk, all
in source big-endian order. The temporary PREPARE and indexed SELECT work
and result slots live only for their owning stages. The wrapper emits
PREPARE, signed SELECT, and PREPARE adjoint. The zero-signal preparation
for the resulting block is identity. Structured or repeated outer
controls use the existing selector and predicate owners.

The original tutorial instance is `(T - H + X - Z) / 4`, with one-bit
coefficient precision and port widths `(1, 2, 5)`. Its original formal
fixture and the user-facing native notebook share the same term order,
coefficients, signal metadata and resource model. The reference checks
walk the unchanged source graph using stage-local bindings and compare
the complete ordered primitive stream on each physical wire, with
operands and classical feedback. Both directions match the full complex
reference over all 256 measured histories each for the coherent test
input. All temporary work qubits return to zero in every tested history.

| Count | Physical emission | Independent source model |
| --- | --- | --- |
| T equivalents or T/T-dagger | 60 T/T-dagger | 48 T equivalents at the source default rates: four Fredkins and eight And calls |
| Measurements | 8 | 8 |
| Other model categories | Controlled T/H/X/Z appear as native controlled opcode leaves | 66 Cliffords and 3 rotations |
| Wires | 10 reserved, including two temporary wires | 10 peak logical qubits |

## Original test coverage

| Source tests | Native evidence |
| --- | --- |
| Default fixture, signature, normalization, signal state and tensors | Original four-term catalogue and exact source stream/full-state comparisons. Existing normalized coefficient PREPARE, invalid admission, zero signal and model tests cover the owners. |
| Metadata, error and resource tests | Existing mixed auxiliary-size and nonuniform-normalization tests preserve the implemented maximum normalized error rule. Existing directional, controlled, and legacy-profile resource tests cover the specialized calls. |
| Exact signed coefficient families and Hermitian checks | Existing native 2/3-term signed cases, non-Hermitian T/S phase cases, and the exact original four-term example. The complete original 2/3/5-term operand and coefficient combinations remain open. |
| Approximate and random tests | Original source tests execute unchanged. Accepted native alias tests cover concrete one- and eight-bit examples. The original nine-bit and random dense fixture families remain open natively. |
| Notebook | The frozen 10-cell original and the native 12-cell notebook execute independently. The native notebook retains the original signature and call graph and adds a complete complex system projection and work check. |

The unchanged non-notebook source run passes 59 cases with one mode skip
and one expected serialization failure. Source external-object and
returned-adder failures remain documented in the control-owner audits.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_linear_combination.ipynb
```
