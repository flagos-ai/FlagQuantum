# Cast source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Frozen bookkeeping files
are Apache-2.0, copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0320 | `cast.ipynb` | `9895dcc29482ecf980c50d16af4776f309e4202c` | Closed by the complete native tutorial. |
| Q-0321 | `cast.py` | `9fd7bee565a88e97a4aa1b1c7945c5c6529d4381` | Actual source factory foreign resolution and direct SDK views verified. Classical word views and full executable dtype admission remain open. |
| Q-0322 | `cast_test.py` | `c018d2af7aa811b11ff75753927ef19a1da85ecb` | Closed for the runnable original scenarios and notebook. |

Compiler Cast retains the original bit pattern. Bind the left and right
word on the same underlying wires, with big-endian bits and row-major
array indices. Concrete typed Program paths reuse split/join rebinding.
Simulation owns the encode-then-decode numerical callbacks. The original
32-bit QInt to unsigned QFxp example keeps the exact wire order, verifies
both signed extrema and the zero and unit cases, and presents the original
directional signature. The typed Program verifier admits a smaller set of
physical descriptors than the metadata and numerical codecs.

In the original four-bit cast–add–cast wrapper, both Cast stages emit zero
gates. The Add accounts for all 12 T/T-dagger instructions, three
measurements and three caller-owned clean work bits. The exact source
stream, operands and feedback agree. Both source QEC profiles match
independently, including the eleven-qubit logical peak. Every one of the
eight histories matches the locked tensor map on all components of a
coherent input and returns the work bits to zero. The original eight-bit
classical sample gives the fixed-point value `9/256`. Bit-level mixed
storage opt-in is covered as a representation operation.

Upstream's adjoint drops shaped layout and mixed-storage opt-in. Native
metadata adjoint retains both, and tests preserve the locked raw source
behavior as a diagnostic. The upstream shaped tensor path also raises
for array-valued connections. Native typed array rebinding keeps physical
order and does not change amplitude. Mixed storage metadata does not
authorize a coherent Program conversion or replace explicit measurement.

The original non-notebook run has six passes and three fixture-mode skips.
The original eight-cell and native ten-cell notebooks pass independent
kernels. The native tutorial retains the original example and signature,
then checks a small coherent circuit through the normal append entry.

The shared typed signature retains both descriptors, the paired `reg`
ports, and the declared shape. The source-derived cases cover signed
fixed point, symbolic input width, classical storage, and both explicitly
enabled mixed directions. The locked width check permits a symbolic
input with a concrete output. A concrete input with unequal symbolic
output fails. Zero-width classical metadata retains its storage domain.
Independent source qubit counts agree, including the all-classical peak
of zero.

An owned CBit cast returns its existing slot with no cast instruction.
A two-element fixture preserves all four classical inputs and both slot
identities. The measured fixture uses four actual Cast calls around the
reordered outputs of two MeasureZ calls. Through append it emits exactly
the two measurements. All four histories match the frozen instrument
on complete complex amplitudes and return the clean work bit to zero.
The source model separately counts the two measurements and zero T.
Mixed-domain declarations remain metadata. Full executable dtype admission
and classical word views remain open. Actual source factory transport and
SDK consumers are detailed below.


## Actual factory transport and approved fixed-point execution

The original `_cast` QInt(32) to unsigned QFxp(32,32) factory and a shaped
four-bit QAny to QFxp fixture preserve exact native library bytes and reconstruct
equal in the unmodified source at depths 0, 1 and 2. Existing explicit native
factories consume the already-decoded descriptor arguments and build the
existing cast function. Original array-valued shape arguments remain arrays at
the interchange boundary; the typed signature supplies the canonical shape.

The actual 32-bit factory binds reversed physical words with no primitive,
measurement, allocation or workspace. Its source SDK hook preserves the exact
array object, and its PennyLane hook returns None. The independent source QEC
model is zero and peak width is 32. This case verifies complete wire ownership
and zero-operation identity without allocating a 32-qubit statevector. The
shaped eight-data-bit fixture adds an entangled spectator and checks every
complex amplitude through ordinary append. Its source model independently
reports zero QEC cost and peak eight.

Two historical Cast/Add/Cast fixture assertions still expected the unsigned
fixed-point rejection predating approved Proposal 115. Both failures reproduce
on the unmodified batch base. The typed fixture now retains its original QFxp
port, replacing the old opaque substitute, and its two Cast functions retain
actual nominal descriptors. One- and four-bit wrappers match every original
primitive, operand and feedback. Every small measurement history matches the
full unnormalized source complex map with all clean work zero. The existing
uncontrolled append path and declaration-composition consumers also pass.
Earlier source gate inventories and independent model boundary diagnostics
remain separately recorded above.

The unmodified source CUInt(4) to CInt(4) view reinterprets all sixteen raw
words correctly and its SDK hook returns the same storage object. Source
serialization raises its original unsupported-CUInt TypeError at depths
0, 1 and 2. Native owned classical word casts are a separate missing capability;
existing `cast_word_function` rejects their descriptors. Proposal 118 records
an explicit protected contract for slot-preserving classical reinterprets and
requires approval before implementation. Quantum/classical mixed-domain casts
remain declaration-only. Full signed/empty fixed-point physical storage remains
under pending Proposal 116. Q-0321 therefore stays open.

Receipts `g2-cast-foreign-*` record complete factory transport, baseline failures,
strict changed-file checks, concentrated/integration/dual-Cirq regressions,
frozen source hashes and unchanged execution owners. No source patch or
physical admission change is part of this batch.
