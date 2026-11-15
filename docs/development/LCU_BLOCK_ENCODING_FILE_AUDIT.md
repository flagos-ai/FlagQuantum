# SELECT and LCU block encoding source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The frozen files under
`qualtran/bloqs/block_encoding/` are Apache-2.0, copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0282 | `lcu_block_encoding.ipynb` | `4f457c91bedc3baaf94f7515242a67fffc4bf4b7` | Closed by the native tutorial. |
| Q-0283 | `lcu_block_encoding.py` | `679ac7964baf62288713078a5ec4dbce674cbece` | Open for generic source external objects, register typing, and wire-symbol dispatch. |
| Q-0284 | `lcu_block_encoding_test.py` | `0632799611bbd5ec2b7ca724533d4bbd1bbb7169` | Closed by the original-fixture mapping and existing controlled branch evidence. |

## Binding and work ownership

SELECT-only exposes PREPARE as the signal state and emits one SELECT. The
LCU wrapper emits PREPARE, directional SELECT, then PREPARE adjoint, with an
identity zero-signal preparation. Specialized control changes SELECT alone.
The Hubbard catalogue binds the existing PREPARE and SELECT direction
callbacks. Packed registers use the existing partition helper in source
order. This regrouping emits zero gates. The named and packed forms give
identical physical instructions, operands and classical conditions at
the same binding.

For the original 3 × 3 parameters, the selection, preparation junk, and
system widths are 12, 2, and 18. Both representations have normalization 72
and source error metadata zero. PREPARE owns the two logical direction
bits. SELECT does not receive those bits. Each stage owns its temporary
work and result slots during the call. The sequential stages can reuse the
same physical allocation.

## Original scenarios

| Original tests | Native counterpart |
| --- | --- |
| `test_lcu_block_encoding`, `test_black_box_lcu_block_encoding` | Original 3 × 3 metadata, ordered parameters and width, one PREPARE/SELECT/PREPARE-adjoint each, independent QEC/peak model, emitted CircuitIR codec round trip, exact named and packed streams for both directions and both control polarities. |
| `test_select_block_encoding`, `test_black_box_select_block_encoding` | Same original Hubbard metadata, one modeled and executed SELECT, exact named and packed emission in both directions and emitted CircuitIR codec round trip. The preparation remains available as the independent signal state. |
| `test_ctrl_lcu_be_cost` | Native positive and negative control models match the unchanged source. The positive case declares 28 Fredkins, 77 And operations, 294 Cliffords, 16 rotations and 77 measurements. |
| `test_notebook` | The frozen 40-cell notebook and the 42-cell native tutorial independently execute. The native tutorial contains eight signature and call graph figures and checks the compiled positive-control operation. |

The original non-notebook run has 17 passes and four mode skips for
decomposition-derived counts. The native catalogue matches the source fixture
metadata and modeled costs at 3 × 3. Existing unit tests compare original
PREPARE and SELECT decomposition details and all small SELECT/LCU composite
histories. The separate packed Hubbard 2 × 2 trial compares a complex
projected state with the independent Hamiltonian action and checks every
temporary bit in four representative measurement histories. Full 3 × 3
state execution and exhaustive Hubbard histories remain unverified. The
codec verification applies to emitted native IR.

## Emission and source model

| Original size | Emitted T/T-dagger, measurements, reserved wires | Source Fredkins, And operations, measurements, peak logical qubits |
| --- | --- | --- |
| 3 × 3, no control | 540, 86, 38 | 28, 74, 74, 37 |
| 3 × 3, positive control | 552, 89, 39 | 28, 77, 77, 39 |

The physical Hubbard stage expands its bounded arithmetic and Fredkin
decompositions. The locked source model uses abstract charges for those
children. Both columns retain their own counting and scheduling conventions.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_lcu_block_encoding.ipynb
```
