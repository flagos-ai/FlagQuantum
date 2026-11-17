# Allocate and Free source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The frozen files under
`qualtran/bloqs/bookkeeping/` are Apache-2.0, copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0310 | `allocate.ipynb` | `8d91513e62d1015fcbd97c0601442916e3ba6796` | Closed by the complete native Allocate and Free tutorial. |
| Q-0311 | `allocate.py` | `e459095285de98c1468b661afdbcbe58238ff1ad` | Open for external object hooks, full executable dtype and symbolic component drawing. |
| Q-0312 | `allocate_test.py` | `96f411955b10a466d37e8ba3b0698a0823d42ba9` | Closed for the runnable source scenarios and notebook. |
| Q-0323 | `free.py` | `8a0c73db71e1d8ffa31b2c7a15d246f02f8f4f63` | Open for external object hooks, full executable dtype and symbolic component drawing. |
| Q-0324 | `free_test.py` | `63ffa89314718cfaa94bfb0104953e52fd1c518e` | Closed for the runnable symbolic fixture and ten-wire utility round trip. |
| Q-0309 | `_bookkeeping_bloq.py` | `32b9b7498417c20ee8ca86c8e53f16b2d63bada4` | All ten descendants verified on admitted concrete paths. Generic descriptors remain open. |

## Lifetime, bit order and ownership

Compiler Program records own the typed operations and physical binding. Allocate
creates a right-side `reg` word and Free consumes a left-side `reg` word.
The adjoint exchanges their roles while preserving the dirty flag. Packed words
retain big-endian order and physical bindings use the caller's pool order.
Clean work comes from the clean pool; borrowed work comes from the separate
borrowed pool. The allocator tracks provenance, so a mismatched declaration
at release does not transfer ownership to the other pool. Numerical callers
must restore clean wires to zero and restore borrowed wires with their joint
state intact. Control passthrough does not guard either lifetime operation.

The classical source allocation decodes an all-zero word under both flags.
Classical Free rejects nonzero clean words and accepts nonzero dirty words.
The source tensor views contract a zero ket or bra even with the dirty flag.
Native numerical pool binding preserves the live borrowed state according to
the caller contract. The tensor zero projection is a separate diagnostic;
its amplitude is intentionally unnormalized. Existing Cirq probes compare
ordered qalloc, qborrow and qfree events with native leases. The source
PennyLane hooks return no operation. Native emitted circuits likewise have
no lifetime gate. Direct PennyLane and Cirq object-level bookkeeping adapters remain open.

Concrete typed component scores identify lifetime calls by their primitive
bodies. Allocate and Free reuse the existing default arrows and empty
connectors. The glyphs and full layout match the frozen scores for one
and three bits under both dirty flags. The tested source `qalloc` or
`qborrow` and `qfree` calls match the native lease events on the same
reversed pool. Both bindings produce zero gates. The physical circuit
reserves the pool and one spectator. The source qubit model visits its
disconnected spectator separately and reports the pool width. The
locked symbolic glyph has the same labels, while native typed component
scores require concrete widths.

## Shared bookkeeping control

The unchanged base callback returns the original control tuple and invokes
one child with only its data bindings. The native test family covers
Allocate, Free, ordinary Split and Join, Partition in both directions,
Split2, Join2, and Cast. Positive, negative, shaped and grouped controls
exercise one actual native call and one actual source call per case.
New wire-map checks compare the source Cirq output maps for pair rebinding
and shaped Cast. Complete complex states agree on the bound circuits,
including their idle controls. These callbacks emit zero gates and require
no predicate workspace. Allocate and Free retain their existing pool events.

The shaped CBit Cast keeps its two original classical slots. The only
emitted instructions initialize the caller's two input bits. The locked
graph accepts the shaped Cast call, and its numerical callback raises
on the array. Scalar source callbacks provide the value oracle for each
element. The two finite-field polynomial descendants live under
`gf_poly_arithmetic`. Their binary coefficient views also use the shared typed control callback.
Generic descriptor admission remains open in Q-0533.

## Original cases and resource accounting

| Fixture | Native evidence |
| --- | --- |
| Symbolic Allocate and Free | Directional typed signatures, atomic leaf multiplicity one, zero QEC gates, independent source peak `Max(0, n)` under both flags. |
| Allocate, ten H, Free | Ten emitted H on the allocated word. Full contracted scalar is `1/32` and squared norm is `1/1024`, matching the locked reference. |
| Allocate, Split, Join, Free | Original ten-wire identity flow, allocation and release in original order, no output word or emitted gates, full zero state after append. |
| Previous controlled and mixed-flag paths | Reuse the complete gate, allocator ordering, measurement feedback and complex branch evidence in the corresponding workspace lifetime tests. |

The two ten-wire fixtures reserve ten physical wires. The first fixture's
H layer and tensor contraction are distinct from the zero-gate allocation
and free model. The second fixture has no primitive gates or measurements.
The native runner retains both symbolic leaf graphs and both directional
register diagrams. The unchanged 18-cell original and 20-cell native asset
pass independent kernels. The unchanged original tests have six passes and
six mode skips with the notebook executed separately. The mode skips cover
the original atomic decomposition, call-graph comparison and type checking.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_allocate_free.ipynb
```
