# Sparse rotation preparation source audit

The Qualtran v0.7.0 reference is commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. The source and test blobs are
`e5d631263c199129e23c246bfab0fd36fd5f774a` and
`ea6a6748f0837615f45e7e6082188b5cc7cfc1d9`, respectively.

## Ownership and execution

| Source responsibility | FlagQuantum owner |
| --- | --- |
| Ordered indices and coefficients | `SparseRotationPreparation` takes explicit paired tuples. The original array factory inputs are bound in the example catalogue. |
| Inferred or overridden target width | The caller binds the target wires. The original examples reserve four and six wires for the same coefficients. |
| `signature` and partition of the target | Target then gradient, both through and big-endian. `split_register_pair` reserves no scratch and emits no gates. |
| Dense child and normalization | `RotationPreparation` consumes the padded, normalized low-word coefficients and returns the gradient. |
| Partial permutation and join | The join restores the original word order. The shared permutation consumes the complete target and preserves the source cycle traversal. |
| `build_call_graph` | The concrete cost composes the dense child and permutation once each. The symbolic shape formula retains the source's worst-case cycle of length twice the support size. |
| Temporary ownership | The dense preparation and permutation reuse the same clean work pool and classical slots after the dense stage resets its work. |

Run `python -m examples.algorithms.sparse_rotation_preparation_catalog` to
exercise both original fixtures through `circuit.append(op)`.

## Verification

The original two factories retain the coefficient and register order, exact
partition/dense/join/permutation call sequence, gate stream, measurements and
feedback operands. The three original state fixtures (inferred, explicit four
and explicit six target bits) retain the precise input floats used to quantize
angles. Each direct source state contraction agrees with the source's expected
complex amplitudes, including all padded addresses. Three seeded native runs
per fixture check the full target/gradient/work state, including phase-gradient
restoration and zero work. Smaller previously accepted cases exhaust all raw
measurement histories when there are at most five events.

| Target bits | Emitted T/T† | Emitted CX | Emitted Rx | Emitted measures | Reserved wires | Source model And | Source model Clifford | Source model measures | Source model logical wires |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 176 | 322 | 6 | 44 | 12 | 40 | 194 | 40 | 12 |
| 6 | 256 | 442 | 6 | 64 | 14 | 60 | 294 | 60 | 14 |

Source QEC counts treat the phase-gradient addition differently from the
native measured carry lowering. Both original independent source models also
declare zero Toffoli, zero CSwap and zero rotations. The emitted Rx are
retained in the native stream.

## Remaining boundaries

- Q-0697 remains open for general source array/map adapters and executable
  symbolic binding. Direct concrete tuples and independent symbolic cost
  declarations already have native owners.
- The original partial map loses an indexed transition for `(3, 0)`, and
  the original singleton decomposition cannot bind its target. The native
  emitter explicitly rejects both cases. The earlier [component
  audit](REGISTER_SPARSE_ROTATION_PREPARATION.md) records the witnesses.
- The original source test run passed nine cases. Two decomposition dtype
  checks skipped on the source's BQUInt-to-QUInt connection, and two
  serialization cases were expected failures. The native port binds the
  concrete through wires and leaves external source-object serialization
  outside this test migration.
