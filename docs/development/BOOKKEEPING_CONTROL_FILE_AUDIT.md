# Bookkeeping control source-file audit

## Source and ownership

The frozen reference is Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, file
`qualtran/bloqs/bookkeeping/_bookkeeping_bloq.py`, Git blob
`32b9b7498417c20ee8ca86c8e53f16b2d63bada4`.
Original Google Apache-2.0 attribution remains in the native owner.

The complete executable file consists of `_BookkeepingBloq`, its
`get_ctrl_system` method and the nested `add_controlled` callback. The native
owner is `compiler/registers/control/component_bookkeeping.py`. Existing Core
Program records and the shared component builder own signatures, calls, binding
consumption and errors. No parallel source object hierarchy is introduced.

| Original behavior | Native owner and acceptance |
| --- | --- |
| Return the same child object | Factory returns the same ProgramRecord |
| Ignore the optional CtrlSpec | Callback does not construct a predicate |
| Retain the supplied control Sequence | Direct helper returns tuple/list/empty container by identity |
| Invoke `bb.add_t(self, **in_soqs)` once | Existing builder calls the selected child once |
| Preserve output order | Existing child signature and builder tuple order |
| Accept generic child descriptors logically | Existing typed declarations preserve symbolic and supported dtype metadata |

The declaration factory traverses every stored reachable body and rejects
computational, untyped, physical or invalid definitions. Body-free typed
functions express caller-supplied logical declarations. They do not attest to
the semantics of a future implementation. Replacing one requires revalidation.
The existing physical binder rejects a missing body. Concrete execution and
storage admission stay with their existing owners.

## Complete descendant and dependency inventory

The frozen base has ten concrete descendants. Allocate, Free, Split, Join and
Cast are defined in their corresponding bookkeeping files. Partition, Split2
and Join2 inherit through `_PartitionBase` in `partition.py`. GFPolySplit and
GFPolyJoin are defined in `gf_poly_arithmetic/gf_poly_split_and_join.py`.
The base imports Bloq, BloqBuilder and SoquetT, with typing-only control aliases.
Those responsibilities map to existing Program records, typed bindings and the
component builder. This batch changes no physical executor or descriptor.

`test_bookkeeping_control_file.py` invokes the actual unmodified original
callbacks for all ten descendants. Generic logical cases include symbolic
widths, signed and empty fixed-point metadata, classical typed Cast and packed
polynomial coefficients. It compares selected-object identity, the unchanged
control container, untouched input maps, one direct child call, register order,
operand producer/name/index and resource call multiplicity. Tuple, list and
empty controls are exercised. A computational implementation substituted for a
declaration is rejected. The physical missing-body failure is asserted.

## Reused execution and resource evidence

`test_bookkeeping_component_control.py` covers finite bodies, control polarities,
allocation flags, exact source call traces, aliases, ownership errors, complex
joint states and `circuit.append(op)`. `test_field_polynomial_component.py` and
`test_packed_field_polynomial_component.py` cover the two polynomial descendants,
MSB/coefficient order, actual original controls and typed alias execution.
`test_controlled_workspace_lifetimes.py` covers exact adjoints, measured cleanup,
all applicable small branches and restoration. These owners remain unchanged.

The base callback contributes zero primitive gates, measurements and workspace.
Its call graph contains one selected child. Child primitive costs and source
resource models are reported independently by each child owner. Allocate/Free
lifetime effects persist even though the base contributes no gates. Free neither
resets nor tests a state. Borrowed-state restoration and zero-before-clean-free
remain caller obligations. No new numerical protocol is claimed here.

## Independent boundaries

Q-0309 closes the complete base control protocol. Q-0533 continues to track
polynomial-specific foreign, symbolic and nonbinary execution. Source
control accepts arbitrary source dtypes, while native logical construction uses
existing native descriptor contracts and explicit translations in reference
tests. Foreign objects stop at their existing ecosystem boundary. A logical
fixed-point declaration provides no physical admission. Signed and empty
fixed-point storage remains subject to pending Proposal 116. No approval for
that proposal is inferred from the approved Proposal 115.

Unsupported child descriptors, symbolic shape dimensions, physical symbolic
widths and absent child implementations retain their existing owner diagnostics.
The base contains no decomposition, numerical kernel, measurement feedback or
resource-model override to transplant. Those independent child-file obligations
remain open in the ledger.

## Batch verification

The concentrated owner regression passes 332 cases, local integration passes 17,
and the complete base/descendant callback profile passes 199 under Cirq 1.6.1 as
well as the default 1.7.0 environment. Strict typing checks both changed Python
files. Black, Ruff, team scope, architecture, IR and protected API checks pass.
Eight frozen source blobs and six unchanged native owner files are verified.
Qualtran 0.7.0, Kickmix 0.1 and its Simulator are checked before evidence capture.
Whole-repository dependency findings are recorded separately from the zero
changed-surface findings in `g2-bookkeeping-control-file-evidence.json`.

The first concentrated run exposed two diagnostic-text regressions introduced
while editing this owner. The original refusal text was restored and the
unchanged consumer tests pass in the complete rerun. This batch changes no
physical storage contract.
