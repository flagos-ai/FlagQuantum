# Unitary block encoding source files

The shared typed graph, exact child-pair constructor binding and original
deep-wire failures are documented in
[the packed-block constructor audit](RESOLVER_PACKED_BLOCK_AUDIT.md).
The execution and metadata receipts below retain their original scope.

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The frozen files under
`qualtran/bloqs/block_encoding/` are Apache-2.0, copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0305 | `unitary.ipynb` | `cca05e99e365416bada5a7785a43ac1ba3d57555` | Closed by the full native tutorial. |
| Q-0306 | `unitary.py` | `bbcff63d78cbd58979bee3670597a778fdd4bdd2` | Closed with complete native/optional-external binding, shape and symbolic admission, controlled paths and owned IR serialization. Source serialization and partial-controlled-gate failures are retained below. |
| Q-0307 | `unitary_test.py` | `4bead193a26eb24303fb96a15bd2233774924e0e` | Closed for the original runnable scenarios and notebook. |

## Interface, bit order and ownership

The Algorithms wrapper accepts the existing typed signature and admits
concrete scalar quantum through ports for a physical binding. It rejects
input-only and output-only children before emission. The Compiler's shared
partition layout concatenates the child system words in signature order.
Each word is big-endian. The outer signature declares system, ancilla and
resource in that order, with zero-width entries omitted. Ancilla and
resource pass through untouched. Caller-provided work and classical slots
belong to the child's decomposition. The wrapper owns no allocator, gates,
or feedback of its own. The signal state delegates to identity preparation.

The single directional child contributes the direct call and gate model.
Auxiliary ports contribute bystanders to the logical peak. The plain and
overridden T fixtures each emit one T and no measurements, and reserve
one and four physical wires respectively. The independent model declares
one T equivalent for each with logical peaks one and four. The earlier
[execution audit](UNITARY_BLOCK_ENCODING.md) covers 48 model comparisons
across both profiles, complete source gate sequences and all small
measurement histories, including clean work and coherent auxiliary states.

## Original scenarios

The default and overridden T examples preserve the complete packed
signature and the declared scale and error. Their full 2-by-2 and
16-by-16 complex matrices agree with the locked tensor contraction;
the forward stream has T on the system only and the inverse restores
all basis columns. Tests verify identity signal preparation and the
source's refusal to wrap a one-sided integer state. The concrete typed
adapter uses the existing register metadata. The shape cases below
cover which source declarations can bind.

The unchanged original non-notebook run has eight passes and two
expected source serialization failures. The original 11-cell and native
13-cell notebooks both pass independent kernels. The native tutorial
preserves both fixtures, their graphical signatures, and the single-call
graph, then exercises append and adjoint with full complex matrices. The
native notebook runner rejects reference imports and checks both SVGs.

## Shaped, symbolic and controlled bindings

The locked wrapper declares the sum of child element widths, without
multiplying by the register shape. Concrete single-element shapes
`(1,)` and `(1, 1)` bind in the original child port order and agree
with the complete locked matrix. A two-element shape and a symbolic
shape each declare width one and fail locked partition. Native
declarations retain width one and reject the incompatible numerical
binding. Symbolic ancilla and resource widths appear in the packed
signature and the independent child peak formula. They require
concrete sizes before execution.

The generic controlled binding places the directional child behind
the source positive predicate. It reuses the shared negative-control
and measured conjunction paths. The comparison covers 16 configurations
including direct/inverse, four activation patterns, and coherent idle
ancilla/resource values. Every applicable small measurement history
matches the full locked complex output with predicate work at zero.
The two QEC profiles agree with the independent locked model.

For mixed-polarity three-bit control of the T wrapper, native IR emits
one controlled phase, eight T/T-adjoint instructions for the predicate,
and two measurements. The source model counts one rotation, two And
calls and two measurements. A source-compatible quantum child binds
by its named emitter. The optional external snapshot and serialization evidence
below completes the remaining binding obligations. Source object serialization
failures are preserved rather than advertised as compatibility.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_unitary_block_encoding.ipynb
```


## Direct external objects and canonical serialization

The shared optional Ecosystem importer now captures the actual two original
factory objects. Their forward/adjoint and positive-controlled forward/adjoint
operations preserve alpha, epsilon, packed widths and idle logical ports. Every
full complex matrix matches the unchanged source tensor, including the overridden
factory's coherent ancilla/resource values. Each ordinary direction emits exactly
one T/T-adjoint instruction; each positive-controlled direction emits one cphase.
Independent supplied models retain the source gate categories and widths.

Each emitted circuit roundtrips through existing CircuitIR JSON and executes with
the same complete complex matrix. No serialization contract or source resolver
is changed. The pinned source's own two protobuf fixture checks still fail in
NumPy with `Serialization Failed` and an inhomogeneous-array ValueError. Native
IR roundtrip success and upstream object roundtrip failure are separate facts.

External single-element shapes `(1,)` and `(1, 1)` also match all four complete
complex matrices. The source's element-width accounting still undercounts a
multi-element shape, and both its tensor path and the importer refuse that
partition. Unresolved shape/auxiliary sizes require explicit concrete resolution;
existing native declaration and control tests retain those boundaries.

## Directional availability and measured children

The original Unitary(Add(QUInt(2), QUInt(2))) exports ordinary and adjoint circuits
with measured And erasure. Actual imported streams, including their quantum and
classical operands, execute correctly for every small measurement history on a
coherent input spanning all system and nonzero logical auxiliary values. Private
work returns to zero. The same branches agree after canonical IR JSON roundtrip.
T/T-adjoint and measurement counts agree with the independent source Add model;
return-binding swaps are zero in both ordinary directions.

The generic source control reaches Controlled(And), whose partial LEFT/RIGHT
signature has no graph decomposition and whose Cirq one-step decomposition is
None. Both compute and uncompute variants are probed directly. The default
four-way import still reports that unavailable primitive. It does not replace
it with a separately invented controlled circuit.

An explicit `include_controlled=False` request captures just forward/adjoint.
The owned snapshot provider retains the missing controlled pair as None and
refuses controlled binding on that object. This preserves the source's usable
ordinary operation even when its generic control is unavailable. Directional
independent models may likewise be None individually; a known ordinary cost is
usable while an unknown adjoint/control cost remains unavailable. No zero-cost
placeholder or implicit matrix synthesis is used.

| Complete source definition | Native owner / evidence |
| --- | --- |
| Fields and THRU validation | UnitaryBlockEncoding metadata and typed constructor; original one-sided refusal tests |
| system_bitsize and packed signature | Existing signature/partition owner; both original examples, scalar/singleton shapes and symbolic declarations |
| signal_state | Identity BlackBoxPrepare, with original missing normalization behavior retained |
| build_call_graph and build_composite_bloq | One supplied directional child plus idle bystanders; unchanged gate/operand/feedback comparisons and direct external snapshots |
| Controlled/adjoint inherited paths | Existing shared predicate control and supplied positive child; complete matrices/all small histories, original Controlled(And) refusal recorded separately |
| Display and example factories | Full native tutorial preserves both source factory signatures and the single-call graph; native representation uses owned component identities |
| Serialization | Canonical native IR roundtrip; original source protobuf failures remain source failures |

All source-file obligations are covered within those explicit source and import
admission boundaries. This does not claim universal synthesis of arbitrary
foreign gates or repair the pinned source's partial controlled operation.

Validation: 295 concentrated cases pass in 76.34 seconds, including all existing
Unitary tests, actual external factories, measured children and LCU consumers.
The native Unitary notebook passes its independent kernel. Both changed
production owners pass strict typing; Black/Ruff, scopes, architecture, IR and
public API checks pass. All three original Unitary source assets match their
frozen Git blobs. No reference file, stable schema or resource convention changed.
