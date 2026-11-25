# Native phase reads of signed and binary-scaled quantum numbers

The authorized native-example migration extends unreleased `phase(expr, angle)`
capture to arithmetic over initialized signed and binary-scaled `QNum` ports.
The meaning is `exp(i * angle * expr)` on each computational basis value,
including negative two's-complement values and exact binary fractional scales.
Metadata continues to use existing `Annotated[QNum, (width, signed, fraction)]`.
There is no new export, signature, public schema, IR type or opcode.

The existing shared `expand_phase` kernel is authoritative. The native frontend
supplies its existing `signed_word` and `fraction_bits` callbacks, rather than
inventing a numeric evaluator or treating a formatted word as an unsigned raw
integer. Preserve every read witness even after cancellation. Keep existing
24-input-bit, degree-eight, 256-monomial, product-count, finite-coefficient and
actual circuit/workspace/memory limits. Numeric phases do not enter the Boolean
factor fallback until that path propagates formats correctly; overflow fails
closed. No silent unsigned or unscaled fallback is permitted.

This admission covers unguarded arithmetic in root program phase expressions.
It does not admit numeric conditions, numeric comparisons in phase expressions,
numeric ports in helper calls, indexed numeric-word views, or other previously
rejected numeric reads. Unsigned and array phase behavior remains unchanged.
The finite phase coefficient remains classical. Signed/scaled words are read,
never modified, by the phase operation.

The first complete source paths are `qpe.qmod` and `qpe_flexible.qmod` from the
open library. These identical concrete sources evolve a commuting diagonal
two-bit Hamiltonian. Express their controlled diagonal evolution as one exact
phase polynomial followed by inverse QFT, preserving the fractional two-bit
phase output. This is a concrete native QPE composition, not an admission of
general quantum callbacks or arbitrary-unitary QPE.

Acceptance compares all signed/scaled basis phases, coherent complex amplitudes,
cross-word products, constant global phase, cancellation reads, format decoding,
zero workspace and resource rejection against independent Circuits. The two
untouched originals and native counterparts must agree in complete amplitudes
and the joint fractional readout. Preserve all 216 frozen source identities,
replay outcomes and original resource baselines; complete Notebook workflows
remain separate.
