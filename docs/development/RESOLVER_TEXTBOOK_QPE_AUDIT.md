# Textbook phase estimation constructor disposition

Reference: Qualtran v0.7.0 `bcb17bb25360ed68ce3bc084645db3e929799389`,
complete `phase_estimation/text_book_qpe.py` and its unchanged test file.
The frozen Git blobs are `9349ca32908fbba054cab61f095852f880960b0f`
and `39a0d095da29abccf3f9c6d7bf8d48d54a8a8f59`.

## Interface and ownership

The shared Compiler phase owner composes an explicit preparation Program,
one controlled-power Program per phase bit, and a supplied Fourier Program.
It does not select a backend or execute quantum numerical kernels. The existing
Algorithms `textbook_z_phase_estimation_program` delegates to this owner;
Ecosystem tracks native constructor recipes for the four original examples.
Unknown same-signature unitary, window and Fourier children are refused rather
than assigned inferred semantics. No foreign SDK object enters Compiler.

The phase word is named `qpe_reg`, with unsigned fixed-point type and MSB-first
bits. Preparation transfers this RIGHT output to the caller. The unitary target
remains caller-owned. Controls run from the least significant phase bit with
powers 1, 2, 4, and so on. The default Fourier child is the exact textbook inverse
with final reversal. An explicit forward Fourier child or disabled reversal is
honored. Symbolic phase extents retain logical signatures and original formulas;
physical binding still requires a finite executable body.

This source path invokes `BloqAsCirqGate.__pow__`, which retains Power calls.
Bare ZPow exponent fusion does not apply here. The implementation preserves the
unfused controlled calls, their operands and measured predicate feedback.
The inverse reverses typed calls with explicit preparation, power and Fourier
inverse pairs. It does not reverse measurement instructions.

Inverse LEFT phase storage is consumed under the prepared-image clean promise.
It is not a physical projection on arbitrary input states. Forward-plus-inverse
checks retain the entire complex state, including the phase word and scratch;
nonzero phase components are never discarded to make cleanup pass.

## Acceptance evidence

All four original concrete/symbolic constructors resolve in both directions
at serialization depths zero and one, retaining exact register metadata.
Concrete widths one through three, both reversal settings and both Fourier
directions match original root and recursively comparable dependency calls, named operands
and full complex isometries. Native atomic gate leaves stop recursive call
comparison; their accepted primitive/decomposition and numerical evidence is
reused rather than asserting a nonexistent source decomposition. Exact QPE
followed by its inverse restores a coherent target and all zero phase/scratch.
Every raw combined measurement history is executed for widths one and two;
width three additionally uses the independent history certificate and complete
basis-column comparison for its forward isometry, with the zero history for
the roundtrip. This is not a claim to exhaustive large-width inverse channels.

Source-free resolution, SDK-free binding/execution, and ordinary
`circuit.append(op)` retain the emitted instruction payload. Original generic
QPE tests and accepted window/QFT/Power evidence remain separate dependencies.
Current batch receipts in `ftqc-reference-env/g2-textbook-qpe-constructor-*`
record 156 concentrated passes and 145 local integration passes. The unchanged
original ZPow test has three passes; seven original walk/GQSP cases are explicitly
not rerun in this constructor batch. Their existing file-audit dispositions
are retained, including missing dependencies and pending scalar Proposal 119.

## Resources and remaining domains

The resource receipt records 48 directional configurations. Actual emitted
primitive counts and physical width are separate from the original QEC gate
model and logical peak. For the default three-bit forward example, emitted
counts are 40 T-family gates, 60 CX, 36 H, 10 phase gates, 10 measurements
and five physical wires. The source model instead charges one T, ten And,
19 Clifford, seven rotations, ten measurements and five logical peak qubits. Phase outputs are logical outputs, not private work.
The one extra physical scratch bit returns zero in the numerical checks.
Allocation lease release alone is not used as evidence of zero amplitude.

Arbitrary unitary/window constructor contexts, unknown exact adjoints, Kaiser
atomic preparation, LP consumers requiring pending Proposal 119, and symbolic
physical execution remain open. No generic constructor catalogue increment is
claimed for this supported original-example group. Pending Proposal 116, wider
source/refusal boundaries, whole-repository, GPU and performance gates retain
separate scope. Strict typing on all four changed Python files, Black/Ruff, Compiler/Algorithms/
Ecosystem scopes, language, architecture, IR/API and diff checks pass. The
dependency checker retains the identical 720-finding baseline. The reference
source is unchanged.
