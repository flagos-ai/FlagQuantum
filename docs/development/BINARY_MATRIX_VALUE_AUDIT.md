# Complete binary matrix value dependency

## Source and ownership

The immutable Kickmix revision is
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`, Apache-2.0.
Complete source parents K-0306, K-0307 and K-0308 cover
`util/gf2_matrix.cc`, `gf2_matrix.h` and `gf2_matrix.test.cc`.
The committed observation fixture records their Git blobs and five direct
polynomial/fixed-width/word-operation dependencies. The sources remain unchanged.

Core `_binary_matrix.py` owns private, immutable binary matrix value semantics.
A tuple of integer rows and an explicit column count retain empty rectangular
shapes, coefficient order and dimension-sensitive equality. Existing register
metadata and IR remain authoritative. The internal value object has no stable
export, serialized vocabulary or user construction requirement.

Compiler retains circuit construction. Its existing `_field_plu` now obtains
source factors from the shared value owner, converting them back to the existing
dense factors and permutation convention. Its `_inverse_columns` reuses the
shared Gauss-Jordan inverse on the transpose. Both consumers retain existing
signatures, errors and gate emission. Field construction can consume the same
neutral algebra without depending on Compiler or a simulation array library.

## Complete source mapping

| Source definition | Native ownership and behavior |
| --- | --- |
| Default and shaped constructors, rows/cols/row_words/data | `BinaryMatrix`, explicit shape and immutable integer rows; row_words retains ceil(columns/64) |
| identity | `identity`, including the empty identity |
| row pointers | `packed_rows` integer values; coefficient writes return a new matrix |
| get/set/xor_bit | Corresponding native methods preserve coordinates and updates |
| xor_row_into/swap_rows | Corresponding pure row operations, including self aliases |
| is_row_zero | Integer row equality with zero |
| equality/inequality | Dataclass value equality includes both shape and packed rows |
| matrix product | `multiply`, ascending set coefficients and row XOR accumulation |
| polynomial-vector product | `apply`, exact parity, ignoring coefficients beyond the declared columns and retaining the 1024-coordinate dimension bound |
| transposed | `transposed`, ascending set coefficients, empty shape preserved |
| plu_decompose | `plu`, first available pivot, below-diagonal multipliers retained, P*L*U orientation, None on rectangular or singular input |
| inverse | `inverse`, original augmented Gauss-Jordan ordering and empty-matrix failure sentinel |
| rank | `rank`, original rectangular column/pivot/elimination traversal |
| str and ostream insertion | `str(matrix)`, original dots/ones and row newlines |

Source PLU failure leaves output references unspecified. Native failure is an
explicit None result. The existing Compiler boundary preserves its StopIteration
signal and its downstream domain-specific ValueError. C++ mutable aliases are
represented by new owned values. Unsupported C++ out-of-range memory access has
an explicit native ValueError. These adaptations remain internal.

## Exact source observations

`tests/unit/fixtures/generate_binary_matrix_values.py` compiles unchanged
`gf2_matrix.cc`, `gf2_poly.cc` and `fixed_width_int.cc` with a small I/O driver.
It validates all eight source/dependency blobs against the immutable inventory.
Each of 565 inputs records a SHA256 over all original outputs for rank, physical
row-word width, complete string rendering, transpose, inverse, all three PLU
factors, matrix product, polynomial-vector product, sequential row/bit edits,
self aliases and equality. Failure sentinels and dimension errors are included.

The inputs exhaust every square binary matrix through three coordinates. They
also include rectangular and empty shapes, 63/64/65-bit boundaries, 130- and
512-coordinate dense matrices, independently constructed invertible matrices,
mismatched products and the 1024/1025-coordinate polynomial-vector boundary.
Native results compare the complete original output digest. Successful PLU and
inverse outputs also reconstruct both the original matrix and its identity.

All 16 tests in the complete unchanged `gf2_matrix.test.cc` pass with original
`test.test.cc` helpers and GoogleTest 1.14.0 commit
`f8d7d77c06936315286eb55f8de22cd23c188571`. The original compile retains the
GoogleTest char8_t conversion warning. Original result and compile logs are
`ftqc-reference-env/g2-binary-matrix-original-20261122.log` and
`g2-binary-matrix-original-build-20261122.log`.

```sh
python tests/unit/fixtures/generate_binary_matrix_values.py \
  --source-root ../ftqc-reference-env/kickmix \
  --output tests/unit/fixtures/binary_matrix_values.json
python -m pytest tests/unit/test_binary_matrix_values.py \
  tests/unit/test_kickmix_binary_linear_map_file.py -q
```

## Normal execution, counts and models

The existing `Circuit.append(FieldOperation(...))` path executes the same shared
linear-map synthesis. A direct execution-path test records calls to the shared
PLU and inverse methods, compares every complex amplitude with an independently
computed basis permutation, retains an entangled spectator, and restores the
complete state after the exact inverse. Existing strided physical-map, KMX block
mark, resource, field multiplication/inverse/division and measured-erasure
regressions are rerun. The existing 58-case source linear-map fixture compares
complete primitive/mark streams, counts and refusals through width 256.

The matrix value layer emits zero quantum primitives. For the four-coordinate
matrix with packed rows (9,4,5,6), normal fixed-wire forward synthesis emits five
CX and one SWAP. Its inverse emits six CX and two SWAP. Both use zero work and
measurements. The source analytical PLU model separately declares five Clifford
operations and four logical qubits for the forward matrix, with the output
permutation free in that model. The four-coordinate identity emits zero gates
and its analytical Clifford count is zero. These quantities retain their
individual accounting bases.

## Remaining group

Field construction K-0303 through K-0305 and the cyclotomic-factor dependency
remain open. Complete Python GF bindings K-0226 through K-0228 remain open.
FixedWidthInt/word_ops whole-file acceptance is separate from their verified
matrix dependency edges. Pending Proposals 116 through 119 and earlier original
source defects retain their boundaries. G2 acceptance remains incomplete.

## Batch result

479 concentrated regression tests and 29 local integration tests pass. Strict
mypy passes for all five changed Python files. Black, Ruff, Integration scope,
architecture, internal IR, public API snapshot and whitespace checks pass.
The accepted frozen Kickmix 0.1 extension with Simulator remains the reference
execution environment. Existing original field/generator defect boundaries are
retained. The source matrix utility itself has no uncovered definition in this
batch mapping.
