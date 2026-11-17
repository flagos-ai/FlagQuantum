# Register CCX and AND binding

K-0191 through K-0193 cover the locked CCX binding and its two AND lifecycle
entries. The implementation, declaration and original test blobs are
`141dc72e4659109e63ae36160b777bbc3f426461`,
`eddd895569107d1a72a6e82d75f6104c151c4988` and
`3ec89a795efa458dce2208a5979214adb5b969ff`.

The compiler resolves three scalar or array inputs before emission.
The two controls accept ordinary Boolean values, classical bits or
quantum wires. Targets are quantum wires. The native lane order covers
the homogeneous fast paths and heterogeneous fallback. Quantum CCX
keeps the source operand order, and classical or literal controls select
the reduced guarded CX or X schedule. Effective conjunctions preserve
the control semantics of the source's push and pop scopes.

AND initialization performs an X-basis reset before computing each lane,
including on targets initially holding one. AND deletion measures and
clears each target in the X basis, applies phase feedback and then reuses
the same result slot for the next lane. The caller owns that slot. The
native boundary rejects reusing any input control slot for this result,
even when the affected control belongs to a later lane. In the source,
the internal result allocator provides a separate slot.

The 1,020 scalar and broadcast stream comparisons cover all three
operations, including reset placement, ordered guard operands and
measurement feedback. The original tests run unchanged against the
locked extension. A two-lane append test exhausts sixteen reset/erase
histories for each classical input combination. Both work wires return
to zero. For dirty input `|11⟩`, discarded reset outcomes `r0,r1`
produce the exact sign `(-1)**(r0+r1)` observed in the source's reset
instrument. Erasure outcomes add no residual phase.

The mixed two-lane schedule emits two R, one guarded CX, one guarded X,
two HMR, one guarded Z and one guarded NEG at the KMX boundary. The
native IR expands R to H/reset and HMR to H/measurement/conditional X.
There is one reused result slot and no additional quantum scratch.
The source tests report operational non-Clifford counts. The binding
declares no separate analytical cost model.

Run `python -m pytest -q tests/unit/test_kickmix_ccx_binding_file.py`.
