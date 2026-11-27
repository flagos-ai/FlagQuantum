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

## Coefficient input factories

The map factory preserves insertion order for both support and coefficients.
The vector factory accepts one-dimensional numeric sequences and NumPy
arrays, uses the vector length as the domain, and keeps nonzero entries
in ascending address order. Inferred target width uses the declared domain.
The original map factories and the inferred and explicit-width vector
fixtures now reach the native append entry through these factories.

The dense child uses the locked NumPy padding and L2 normalization order.
Float32, complex64 and complex128 vector fixtures match the reference
coefficient precision, normalized child coefficients and gate operands.
When the reference norm underflows or overflows, the existing stable
native normalization can still handle finite coefficients.

The locked rank-two row factory combines the outer length of one with
column addresses beyond that domain. The native factory rejects the
malformed shape at entry. SciPy CSR input raises an ambiguous-length
error before sparse conversion in the reference and native factory.

## Remaining boundaries

- Q-0697 retains the original container-admission and decomposition failure
  cases. Explicit symbolic resolution and actual external execution are now
  covered in [phase arithmetic and sparse binding](PHASE_ADDITION_INTEROP.md). Independent symbolic cost
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

## Actual external object and symbolic execution closure

The complete constructor/default/signature, `dense_bitsize`, `is_symbolic`,
three coefficient factories, partition, normalized dense child, permutation,
composite body, direct calls and both original factories have native owners
in this audit and its linked binding evidence. `from_n_coeffs` is a symbolic
shape declaration; it must receive concrete coefficient/index data and resolved
widths before execution. The two resolved external fixtures now bind through
owned IR, compare every logical column with the native algorithm and restore
the original input with the actual source adjoint. Source and native resource
models, literal matrix leaves and measured arithmetic are listed separately.
No claim is made that unbound symbolic dimensions can be physically executed.

The source CSR factory fails at `len(coeffs)` before its advertised sparse
conversion; native code preserves that error. This is a verified source
admission boundary, not an unimplemented executable sparse-array algorithm.
The unsupported source singleton and partial permutation witnesses remain
unmodified. Q-0697 remains open for those frozen source defects and the
recorded native boundary differences; no sub-checkbox is added.

The new binding closure passes 551 concentrated dependency cases, 60 CPU/
representation integration cases, strict typing of both touched production
adapters and the shared branch proof, and an independent LCU notebook kernel.
The detailed counts, differences between local-matrix and measured permutation
lowerings, and the source-frame lifetime fix are recorded in the linked audit.


## Faithful full-file acceptance (2026-11-28)

The approved faithful-migration closure rule supersedes the historical open
status above. Q-0697 covers the entire locked source file: constructor and
width defaults, all three factories, symbolic declarations, signature,
partition, normalized dense child, permutation, direct call graph, composite
body and both original examples. This batch reuses the existing implementation
and execution evidence; it does not claim a new sparse algorithm.

The later [source constructor audit](RESOLVER_SPARSE_ROTATION_AUDIT.md)
supersedes the earlier partial-permutation rejection description above. The
literal compiler profile preserves the original `(3, 0)` partial-map behavior;
the independent optimized Algorithms injection has a separate contract.
Singleton and unresolved-dimension source decomposition failures, CSR's early
`len` failure, malformed rank-two input and symbolic serialization failures
remain recorded boundaries. None is repaired in the reference or described as
successful source execution. Symbolic phase alone retains the valid parent
root with a declared dense child, without fabricating a physical recipe.

Existing complete small complex operators, applicable measurement histories,
clean private work, spectator and eight-bit angle/gradient evidence are reused
and covered by the concentrated source/binding regression. Literal constructor
precision preserves float32/complex64 normalization; the stable extreme-norm
Algorithms path remains separate. Atomic AddIntoPhaseGrad model calls are not
reported as primitive decomposition. The recorded 4/6-target source models and
measured native gate inventories remain separate, including allocator and
permutation-lowering differences. No large all-history result is extrapolated
from small-case certificates.


Batch verification: 108 concentrated source/execution cases, 30 independent
Cirq 1.6 cases and 32 local integration cases pass. The two Python files pass
strict typing, Black and Ruff; Algorithms scope, architecture, IR, protected
API baseline, required checks, language, links and whitespace pass. All 42
frozen source/dependency blobs are unchanged. Fourteen actual dependency-absence
cases distinguish optional skips from mandatory propagation of the original
error. The dependency-policy scan still has 570 existing findings in 362 files;
none belongs to the new test consumer. That workflow remains open.
