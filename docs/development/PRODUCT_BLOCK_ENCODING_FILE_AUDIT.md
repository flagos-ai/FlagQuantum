# Product block encoding source files

The source-ordered typed Product graph, shared signature replacement and exact
constituent ownership are documented in
[the typed Product audit](RESOLVER_PRODUCT_BLOCK_AUDIT.md). Original tuple wire
and resource dtype failures retain their independent scope.

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen files under `qualtran/bloqs/block_encoding/` are Apache-2.0,
copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0291 | `product.ipynb` | `727fb156b779618d838b430868681e886a902ca6` | Closed by the complete native tutorial. |
| Q-0292 | `product.py` | `0c806922e9d734f7a2d823c960369c1a443ebb4a` | Complete interface, concrete execution, resolved declarations, external objects and owned IR roundtrip verified. Source symbolic decomposition, unequal resource partition and protobuf failures remain explicit. |
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
These are source admission boundaries, not working numerical execution paths.

## Complete definition and external-object closure

`ProductBlockEncoding.of(*children)` retains the source ordered tuple factory.
The constructor owns its nonempty, equal-system and identity-signal checks.
Its metadata and signature retain source normalization, weighted error, packed
flags, shared signal, and resource widths. `make_on` owns the executable
partition, reverse child order, zero-polarity flag checks and inverse binding.
`source_product_block_signature` owns unresolved declarations; the independent
symbolic resource function owns the source fallback. Partition and child
objects remain native owners rather than copied SDK objects. Source display
labels are represented by the existing graph/catalogue teaching assets.

`test_product_external_file.py` imports the actual original T/H constituents
and substitutes them into the unchanged Product consumer. Four resolved
ancilla pairs, including unequal widths in both orders, match the original
symbolic signature and scalar substitutions. Both directions compare the
independently declared direct source costs, the concrete decomposed costs,
all small measurement histories, complete complex amplitudes on occupied
logical flags and signal, clean private flag work, and native IR JSON
roundtrip. No automatic source symbolic decomposition is claimed: the
unresolved original still raises `DecomposeTypeError`.

Both concrete original root factories are captured through the shared
optional Ecosystem boundary in forward and adjoint directions. All ten
original seeded dense root families are captured in all four ordinary and
positive-controlled directions. Each retains exactly one explicit matrix
leaf per source child; complete matrices and JSON roundtrips match the
unchanged source, including idle logical flags. The nested original three-X
root retains its exact primitive operands. This conversion does not contract
an arbitrary component into a replacement matrix or store source objects in
native metadata. A missing independent root model raises explicitly.

All three original factories fail the unchanged source protobuf check with
`Object arrays cannot be saved when allow_pickle=False`. The source files
and diagnostic are tested unchanged. Native emitted IR serialization is a
separate supported contract, not a claim that the original objects serialize.
Both unequal nonzero resource orders retain the original decomposition and
external-import failure. The source assumes every child restores its shared
resource register; a logical resource is not private scratch, and arbitrary
leaking external children do not acquire a cleanup guarantee.

Direct graph flags use all-one abstract controls; actual decomposition uses
all-zero controls followed by X. Their Clifford totals are compared separately.
For the resolved T/H streams, actual T/Tdg count is four per concrete source
And plus the one T leaf, and actual measurement count equals the concrete
source measurement count. Dense leaves have an explicit primitive count but
no invented Clifford/T estimate. Returned-wire binding swaps remain an
independent snapshot statistic. Measurement certificates cover every admitted
history structurally; small histories are also evaluated numerically.

The final concentrated Product, external snapshot, Unitary, feedback and
measurement-certificate regression passes 192 tests without skips. The
Product notebook passes in an independent kernel (eight code cells, two
SVG outputs). The two touched production/example files pass strict mypy;
Black, Ruff, team scope, architecture, multilevel IR and protected API
checks pass. The three source asset blobs remain unchanged. Existing
accepted source-suite and notebook evidence above is reused rather than
rerunning unchanged upstream failures.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_product_block_encoding.ipynb
```
