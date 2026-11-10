# Indexed scalar control of native Pauli-axis rotations

The authorized native-example migration admits an indexed one-bit condition in
`control(x[i], lambda: R(theta, phi, target))`. This extends the unreleased
captured synthesis profile; it adds no export, signature, IR type or opcode.
Existing scalar controls and bounded comparison controls retain their behavior.

Resolve the condition through the existing checked static array-view resolver.
Exactly one in-bounds location is required. A multi-bit register or slice is
never implicitly interpreted as its low bit. Reuse `qreg.pauli_rotation` with
the resolved `control_index`; its initialization, writable-target, disjoint
register and resource checks remain authoritative. Control storage is read
only. Mutable helpers remain outside this admission. Arbitrary equatorial
axes, runtime indices and approximate angle rounding remain unsupported.

The first complete path is a native counterpart of `linear_pauli_rotations`:
an unsigned six-bit input controls three Y rotations with affine angles.
Compose ordinary `R` calls and statically bounded loops with classical real
angle bindings. Preserve the original slopes, offsets and least-significant
array indexing. Do not introduce a source-specific compiler intrinsic.

Acceptance compares the untouched original, native Python and an independent
block-diagonal Circuit in complete complex amplitudes. Check all 64 input
values, coherent input with a relative phase, indexed spectators, negative
angles, bounds and aliases, uninitialized storage and actual resource limits.
The native path must use zero workspace. Rerun all 216 frozen sources and six
approved revisions, preserving raw sources and historical resource baselines.
Full Notebook workflow acceptance remains separate.
