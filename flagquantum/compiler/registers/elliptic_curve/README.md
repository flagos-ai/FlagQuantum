# Elliptic-curve register composition

This directory owns typed composition of the six original ECAdd stages and
their root. Algorithms owns signatures and numerical admission; Ecosystem owns
foreign constructor fields. Compiler imports neither Algorithms nor SDKs and
does not execute numerical kernels.

Coordinates are MSB-first Montgomery words without an embedded modulus. Stage
one preserves a, b, x, y and creates five logical RIGHT flags, ordered f1, f2,
f3, f4, ctrl. These flags belong to the caller, not private scratch. Child
components own their temporary allocations and measurement feedback.

Each stage receives the authoritative signature and explicit owned child
Programs. `addition_component.py` threads all six stages and their bystanders;
`_word_copy.py` shares the five identical split/CNOT/join loops. Stage two creates
lam, stage five consumes it, and stage six consumes all flags. Calls retain source
order and nominal types without replacing quantum behavior by classical callbacks.
Kaliski and doubling can release and rebind caller storage: physical binding must
explicitly transfer input ownership and use the returned layouts. The root
exposes only a, b, x, y and lam_r, with zero private work on its source promise. Source defects and unsupported boundaries
remain documented in `docs/development/ECC_ADDITION_FILE_AUDIT.md`.

`window_addition_component.py` composes the original five-call point-window
root: Join, clean lookup, curve addition, matching bank erasure and Split.
Its five-word view wrapper preserves source port order and raw Montgomery bits
across nominal lookup/coordinate types, using ten zero-gate Casts. Three optional
junk banks bypass arithmetic and remain owned until erasure consumes them.
Ecosystem supplies the Algorithms-owned table and signatures. The default source
erase leaf remains bodyless; callers may explicitly supply the separately
certified native measured erasure. See
`docs/development/ECC_FIXED_WINDOW_FILE_AUDIT.md` for this distinction.

For a typical change, edit the owning stage and run
`tests/unit/test_elliptic_curve_addition_components.py` with the locked reference
environment. Check typed outputs, recursive primitive operands and feedback,
complex amplitudes, private cleanup and actual versus modeled resources.
For the window root, use `tests/unit/test_elliptic_curve_window_components.py`.
