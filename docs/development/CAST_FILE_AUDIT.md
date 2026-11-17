# Cast source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Frozen bookkeeping files
are Apache-2.0, copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0320 | `cast.ipynb` | `9895dcc29482ecf980c50d16af4776f309e4202c` | Closed by the complete native tutorial. |
| Q-0321 | `cast.py` | `9fd7bee565a88e97a4aa1b1c7945c5c6529d4381` | Open for full typed dtype admission and direct foreign-object protocols. |
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
Mixed-domain declarations remain metadata. Full executable dtype
admission and direct foreign-object interoperability remain open.
