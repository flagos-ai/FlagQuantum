# Block encoding interface source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen files under `qualtran/bloqs/block_encoding/` are Apache-2.0,
copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0275 | `__init__.py` | `03c07b29829d3d6322c90faef4508f404e44cef1` | Open. Inventory mapped below. |
| Q-0276 | `block_encoding.ipynb` | `b3f096d4123dd67e39f2d051b98256e868981e4c` | Closed by the native interface tutorial. |
| Q-0277 | `block_encoding_base.py` | `8e6acb38828863e959c0a8543d79bebc0bda1d2f` | Open for symbolic width composition and the generic foreign base-object contract. |
| Q-0278 | `block_encoding_base_test.py` | `534c9e329fdabd21cc005f0ca8ab12148a06110e` | Closed. Its sole scenario is independent execution of the interface notebook. |

## Interface and register ownership

The source interface declares six properties. The existing private
`_BlockMetadata` structural contract has the same six properties for
native concrete composition.

| Source property | Native owner |
| --- | --- |
| `system_bitsize` | Bound block system layout, with the declared child port order. |
| `ancilla_bitsize` | Signal register bound by the selected block implementation. |
| `resource_bitsize` | Additional declared through ports, distinct from temporary child workspace. |
| `alpha`, `epsilon` | Algorithm block metadata, with symbolic scalar expressions for owners that support them. |
| `signal_state` | Existing `BlackBoxPrepare` selection/junk partition and child emitter. |

The abstract upstream base has no gate recipe. The native demonstration uses
the existing unitary wrapper and keeps the independently bound ancilla and
resource qubit untouched. The system is bound to physical wire 1. Identity
signal preparation leaves the clean ancilla on wire 0 in zero. The example
checks the full eight-by-eight complex matrix, the selected two-by-two block
after scaling, and the exact inverse obtained through the same append path.
Its forward resource model declares one Clifford and peak width three.
The forward circuit emits one S and owns no temporary work or classical slots.

The source permits symbolic system, ancilla and resource widths in the base
contract. Native structural metadata currently requires concrete port widths.
The public spectral block protocol has its own existing consumer contract.
Generic source inheritance and foreign-object dispatch remain open.

## Source package exports

| Source export | FlagQuantum status |
| --- | --- |
| `BlockEncoding` | Existing private structural metadata for concrete source-style compositions. |
| `LCUBlockEncoding`, `SelectBlockEncoding` | Domain-owned LCU and SELECT compositions. |
| `LinearCombination` | `LinearCombinationBlockEncoding` in Algorithms. |
| `Phase`, `Product`, `TensorProduct`, `Unitary` | Existing phase, product, tensor product and unitary block owners. |
| `ChebyshevPolynomial`, `ScaledChebyshevPolynomial` | Source-specific executable families still pending. |
| `SparseMatrix`, `SparseMatrixHermitian` | Source-specific block owners still pending. |

The source notebook has four cells. The six-cell native tutorial preserves
the metadata, block projection, normalization, error, SELECT/PREPARE and
literature topics, and adds the executable circuit example. The unchanged
source and native notebooks pass independent kernels. The nine directly affected native suites pass 389 cases without skips.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/block_encoding_interface.ipynb
```
