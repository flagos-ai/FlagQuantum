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
Existing source-compatible word typing admits the five direct ECAdd connections
without supplemental Cast wrappers or numerical conversion. Three optional
junk banks bypass arithmetic and remain owned until erasure consumes them.
Ecosystem supplies the Algorithms-owned table and signatures. The default source
erase leaf remains bodyless; callers may explicitly supply the separately
certified native measured erasure. See
`docs/development/ECC_FIXED_WINDOW_FILE_AUDIT.md` for this distinction.

`phase_estimation_component.py` owns ordered phase-bit preparation, scalar or
shaped window calls, and the original terminal MeasureQFT call.
`double_phase_component.py` owns the two raw coordinate initializations, two
phase calls and two explicitly dirty coordinate frees. Algorithms owns point
powers and numerical readout workflows; Ecosystem supplies the constructor
metadata and typed children. The original terminal has no classical output
and no admitted physical body. These source-shaped roots do not invent a
readout result or silently select the separate executable Algorithms workflow.

For a typical change, edit the owning stage and run
`tests/unit/test_elliptic_curve_addition_components.py` with the locked reference
environment. Check typed outputs, recursive primitive operands and feedback,
complex amplitudes, private cleanup and actual versus modeled resources.
For the window root, use `tests/unit/test_elliptic_curve_window_components.py`.
For the phase roots, use `tests/unit/test_elliptic_curve_phase_components.py`.

`controlled_elliptic_curve_addition_program` threads an explicit outer control
through six already-controlled stage owners. The outer control is separate from
the original exceptional-point flag named ctrl, so stage-local control names
can differ without changing their data or LEFT/RIGHT ownership. Available parent
calls remain stored even when a child genuinely refuses decomposition.
Run `tests/unit/test_ecc_controlled_stages.py` for exact original call operands,
complete reused arithmetic definitions and the actual refusal frontier.
