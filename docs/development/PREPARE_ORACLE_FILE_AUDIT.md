# PrepareOracle source-file audit

Q-0693 covers `state_preparation/prepare_base.py`, frozen Git blob
`6870d7752aeb7da0c251ad1abb3b90519f2d0556` from Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

The source requires selection registers, defaults junk to an empty
tuple, and concatenates selection before junk in its signature.
A derived oracle may supply its L1 normalization. Without an override,
the source normalization accessor raises `NotImplementedError`.
The source documentation descriptor declares the abstract class and
has no factories of its own.

The native consumers bind ordered selection and junk layouts through
the existing preparation emitter. `BlackBoxPrepare` carries explicit
normalization data or an absent value, and exposes the packed bind to
`circuit.append(op)`. The exact source Hubbard implementation verifies
eight ordered selection bits, the two-bit junk word and L1 normalization
32. The source dummy oracle verifies the abstract accessor boundary,
and a selection-only source fixture exercises default empty junk and
signature order. The Hubbard preparation execution evidence is in
[the BlackBoxPrepare audit](BLACK_BOX_PREPARE_FILE_AUDIT.md).

Q-0693 remains open for foreign subclass integration and general
symbolic typed signatures. The current tests establish the composition
contract on the native concrete emitter and distinguish its absent
normalization data from the source exception.
