# Repeated allocation fixtures

Q-0500 and Q-0501 match their inventory blobs at Qualtran v0.7.0
commit `bcb17bb25360ed68ce3bc084645db3e929799389`. The original
structural test passes unchanged. The native fixture composes
existing typed register primitives and workspace binding under tests.

The once mode allocates and releases one clean bit around all
phase actions. The many mode allocates and releases it around each
action. The abstracted mode preserves one nested `inner` call per
input bit, with the allocate and free inside each call. The original
decompositions and their ordered physical gate streams match at
widths 1, 3 and 10. In all modes, the native and original bindings
agree on the caller's one clean slot, release events, and reversed
physical order of the returned logical word.

Each CNOT–T–CNOT step restores the work bit and applies the source
phase to one logical bit. Every full complex vector agrees when
started on the clean-work subspace. Prefix checks verify that the
work bit is zero after every step. The original width model gives
`n+1` for all three schedules. The source QEC model gives `n` T
and `2n` Cliffords. Native emission gives `n` T and `2n` CX with
zero measurements.

The focused native regression passed 140 tests. Strict typing,
formatting, team scope, architecture, IR and public API checks passed.
