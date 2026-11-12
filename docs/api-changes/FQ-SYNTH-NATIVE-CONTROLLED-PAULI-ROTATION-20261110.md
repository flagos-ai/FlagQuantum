# Native controlled Pauli-axis rotations in captured qfuncs

This additive behavior change is authorized for the unreleased synthesis
profile. It does not add an export, change a callable signature, modify Stable
Core, or add an IR opcode.

Inside `control(predicate, lambda: R(theta, phi, target))`, admit only static
Pauli-axis equatorial rotations: `phi` congruent to `0` or `pi` selects X;
`phi` congruent to `pi/2` or `-pi/2` selects Y, with the angle sign adjusted
for the selected axis. The existing compact `qreg.pauli_rotation` verifier,
compiler budget and controlled-Pauli lowering remain authoritative. The
captured effect carries the target bit support and predicate bit as its control.

A comparison remains a one-bit coherent condition. A Boolean combination of
comparison expressions using captured `&`, `|`, `^` or `~` is evaluated through
the existing bounded unsigned expression compiler, limited to eight distinct
input bits. A temporary predicate is computed before the controlled operation,
then uncomputed and released through the existing register lifecycle checks.
Condition and target storage must be initialized, disjoint and writable where
applicable. Classical binding, Python truth operators, arbitrary equatorial
axes, arbitrary callables, and control of mutable helpers remain unsupported.

Acceptance uses the untouched tutorial source, a native qfunc, and an
independent Circuit with complete complex-state comparison, including global
phase and bit order. Include both X and Y axes, both selected/non-selected
control basis states, a coherent control superposition, alias rejection,
unsupported-axis rejection, and resource/workspace accounting. Preserve the
216-source replay and independently rerun the exact original-corpus regression.
