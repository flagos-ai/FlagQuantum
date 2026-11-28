# Double-factorization file audit

The immutable reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, under
`qualtran/bloqs/chemistry/df`.

| Inventory | Source file | Git blob |
| --- | --- | --- |
| Q-0340 | `double_factorization.ipynb` | `1b46c50ef421c01ff48008bc7479787b0ab14b05` |
| Q-0341 | `double_factorization.py` | `e83b385240a5e98b8ab939f6e7eef0fff1547e5c` |
| Q-0342 | `double_factorization_test.py` | `a5a48d8cf2444c21349d61daeed6a4da4d5e2653` |

## Owners and executable boundary

Algorithms owns the one-body and full block resource declarations and the
coefficient-driven block. PREPARE, indexed metadata, programmed rotations and
number phase remain separate component owners. The block uses
`circuit.append(op)`. Persistent signal registers retain their declared
ownership across the block and explicit adjoint. The transient scratch belongs
to the compiler and is cleared after use.

The native tutorial at `examples/notebooks/register_double_factorization.ipynb`
preserves the two original examples with 10 spin orbitals and 250 eigenvalues.
It displays their direction, word widths, array shapes and depth-one graphs,
including leaf multiplicities. The full block score matches the seven ordered
calls and their operand sets in the original decomposition. The middle
reflection uses the outer success bit, the orbital index and spin. A smaller
example constructs and appends an executable block.

Run the tutorial regression with
`python -m pytest tests/integration/test_register_graph_notebook.py -k register_double_factorization`.

## Evidence and resource accounting

- The original standalone suite ran with OpenFermion 1.6.1 and reported 11
  passed and one skipped, with notebook execution selected separately. It
  includes the 108-spin, 360-factor and 13,031-eigenvalue comparison. The native
  model checks the complete ordered source graph at that size.
- The original full notebook model declares 1,245 Toffoli, 65 measurements
  and a 95-qubit peak. The estimate runs with emission disabled.
- The independent coefficient fixture from the prior executable batch emits
  482 T/T† and 89 measurements with four work qubits and 42 reserved wires.
  Its full complex projection is checked against a two-spin fermionic
  reference, including four representative measurement histories and work
  cleanup. Prior tests also cover forward/adjoint recovery on coherent
  nonzero signals. These measurements refer to the executable fixture and
  are reported separately from the large model.

## Remaining source boundary

The source coefficient factory and the one-body and block `alpha` and
`epsilon` properties raise `NotImplementedError`. The supplied cost-only
nodes only declare resources. The source also omits the `l_ne_zero`
predicate in the one-body number-phase and middle-reflection paths
(Qualtran #1022). The native coefficient path implements the intended
controlled behavior; its projection and primitive execution have independent
tests. The actual source constructors accept integer dimensions and precisions.
They do not expose arbitrary external-subcomponent constructor fields. The
remaining physical source gaps and literal full-block comparison are described
below. Q-0341 remains open under the original acceptance requirements.


## Actual top-level graph and model handoff

The complete frozen file contains 31 class/function definitions and the two
`_DF_ONE_BODY` and `_DF_BLOCK_ENCODING` documentation specifications. The two
original examples and their native tutorial remain independently tracked as
Q-0340/Q-0342. This batch adds actual foreign-constructor and stored-graph
consumers using existing native owners. The source implementations retain their
original bytes.

| Complete definition group | Native ownership and verified boundary |
| --- | --- |
| Both classes and their constructor fields | Existing native source-cost queries, owned constructor-library records and explicit model consumers. Both source classes accept integer dimensions and precisions only. |
| Both `control_registers`, `selection_registers`, `junk_registers`, `target_registers`, `signature` | Existing Core register/signature records and Compiler library interchange. Exact ordered registers, sides, array shapes and every connection survive capture and export. |
| Both `ancilla_bitsize`, `resource_bitsize`, `system_bitsize` | Width queries independently checked against actual original register groups. Physical reservations remain separate. |
| Both `signal_state` properties | Existing BlackBoxPrepare/PrepareIdentity capture through Ecosystem's named emitter. Both directions execute owned IR through append, with reversed physical binding, coherent spectator, full complex amplitudes and JSON. |
| Both `alpha` and `epsilon` properties | Actual source calls raise `NotImplementedError`. Their unavailable values remain explicitly recorded. Native coefficient execution is separately tested and does not infer these absent source values. |
| Both `build_composite_bloq` methods | Existing Compiler library resolver and graph exporter preserve exact actual stored graph, occurrence identity, producer/consumer, operand order and shaped indices. One-body has nine calls, full block has seven. Signature-only leaf functions have no executable body. |
| One-body `build_call_graph` | Existing `double_factorization_encoding_model` and direct leaf models preserve the declared counts, actual graph live widths and both cost directions. The fixed declared five-bit phase width remains distinguishable from the actual spatial width. |
| One-body `__str__` | The original label is `B[H_1]`. Native model identities retain `DoubleFactorizationOneBody`. Caller-supplied Drawer labels remain presentation data. |
| Block `build_from_coeffs` | Actual original factory still raises `NotImplementedError` for valid matrix ranks. Existing native coefficient/eigensystem factories have independent fermionic projection and inverse tests. |
| `_df_one_body`, `_df_block_encoding`, both documentation specifications | Existing original example and native notebook/model/score evidence reused. Original import lines remain provenance, with native Algorithms entry points in the contributor guide. |

`test_double_factorization_foreign_encoding.py` exercises actual constructor and
stored libraries at depths zero, one and two, both directions, and six/ten spin
orbitals. All original protobuf fields compare semantically, and owned bytes are
deterministic on another decode/encode. Original resolver reconstruction agrees
for the actual objects and for stored graphs whose topology was exported.
Constructor arguments supply the native model inputs. Native estimation and the
captured signal operation execute with SDK imports blocked after capture.

At depth zero, the original stored CompositeBloq export contains a declaration
shell without connections. Its original resolver reconstructs a different,
empty-topology object. This behavior is recorded as a source boundary. It is not
used as executable evidence. At positive depth, native typed Program export is
exactly equal to the original root entry, including all 45 one-body or 55 block
connections. The original source graph and its adjoint both pass this check.

Signature-only native leaf functions preserve the source's absent bodies.
Core verifies the call graph but physical lowering returns
`unsupported_with_diagnostics`, including when an empty canonical circuit payload
is attached. A resource declaration cannot therefore become an identity circuit
through this path.

## Current physical source gaps

The following remain open for Q-0341 under the original acceptance requirements.

- Original PREPARE and programmed-rotation leaves have analytical declarations
  without primitive bodies. Original Cirq circuit construction succeeds with
  opaque leaves, while decomposition to supported primitive leaves stops at
  those opaque operations. Full top-level original primitive streams and their
  complete measurement histories remain unavailable.
- The original flattened graph contains NumPy uint8 control values that its
  serializer rejects. The existing automatic physical graph importer preserves
  this original error. The unflattened typed-declaration handoff succeeds.
- Original `alpha`, `epsilon` and the signal PrepareIdentity norm property raise
  `NotImplementedError`. The coefficient factory is also unimplemented.
- Original one-body number phase and middle reflection omit `l_ne_zero`. The
  native coefficient path includes the intended predicate. Existing independent
  physics/phase tests verify the native path, while source model and graph
  evidence retain the original wiring.

The declaration handoff and successful native coefficient fixtures do not prove
literal full-block primitive parity. The remaining source failure stages are
captured in `g2-df-encoding-foreign-stages.json`. No source body, serializer,
control or expected result was patched for these comparisons.

## Separate model and measured resource rows

For 50 auxiliaries, 250 eigenvalues, 12 alias bits, eight basis/inner rotation
bits and three outer-rotation bits, both original directions agree with the
following native analytical rows.

| Spin orbitals | Model | Toffoli | CSwap | And | Clifford | Measurement | Rotation | Peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | One-body | 475 | 6 | 14 | 94 | 14 | 0 | 58 |
| 6 | Full block | 1105 | 12 | 64 | 426 | 64 | 4 | 79 |
| 10 | One-body | 574 | 10 | 14 | 94 | 14 | 0 | 79 |
| 10 | Full block | 1304 | 20 | 65 | 431 | 65 | 4 | 101 |

Actual full-source primitive counts are unavailable for these model inputs.
The actual captured signal operation emits two single-wire identity atoms,
zero measurements and zero work in each direction. Its QEC model is empty.
The separate coefficient fixture above retains its measured 482 T/T-dagger,
89 measurements, four clean work wires and 42 reserved physical wires. Its input
data differs from the large model rows. Applicable full-complex, inverse,
measurement-history and work-cleanup tests are included in the concentrated
regression rather than inferred from the analytical rows.

## Validation for the actual graph handoff

The batch passed 237 concentrated cases and 17 local integration cases. The new
foreign-encoding test file passes strict typing, Black and Ruff. Algorithms
scope, architecture, multi-level IR and protected API checks pass. Cirq 1.6.1
passes 75 top-level and foreign-consumer cases. Five source blobs and six
unchanged native owners are recorded in
`g2-df-encoding-foreign-evidence.json`.

The fresh original standalone suite passes 11 tests with one source-owned skip.
The original Notebook test additionally passes in an independent locked kernel.
Its first sandboxed attempt failed at local port binding before notebook code
ran. The separate permitted local-kernel run retains the original notebook and
source bytes and passes. The accepted concentrated regression uses its own
`g2-df-encoding-foreign-accepted-` logs. An earlier interrupted regression and
initial source-protocol assertion failures remain separate diagnostic receipts.


## Complete source disposition (2026-11-28)

The [complete constructor and dependency closure](RESOLVER_DOUBLE_FACTORIZATION_AUDIT.md#complete-factorization-source-closure-2026-11-28) supersedes the earlier missing generic inverse and symbolic declaration paths.
Original absent bodies, unavailable queries and reproduced defects remain
explicit under the approved faithful-source rule. Whole-root physical inventory
and numerical execution remain unavailable. Acceptance requires the batch checks
and verified publication.
