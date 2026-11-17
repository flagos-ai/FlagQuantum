# PREPARE oracle source-file audit

Q-0693 tracks Qualtran v0.7.0
`qualtran/bloqs/state_preparation/prepare_base.py` at commit
`bcb17bb25360ed68ce3bc084645db3e929799389` and Git blob
`6870d7752aeb7da0c251ad1abb3b90519f2d0556`.

The algorithms-internal `PrepareOracle` requires ordered selection
descriptors, defaults retained junk to the empty tuple and caches the
selection-then-junk signature. The signature retains each descriptor,
including symbolic width, shape and direction. The optional coefficient L1
norm raises `NotImplementedError` when an implementation omits it.
The source documentation descriptor declares the abstract class and has
no factories of its own.

Sparse chemistry, THC and Hubbard PREPARE use the shared declaration.
The corresponding walk consumes the coefficient norm through its
preparation oracle. Hubbard retains bounded coordinate descriptors and
the two-bit `temp` word. Sparse and THC retain each existing selector
and QROAM bank in source order. Concrete source comparisons cover
Hubbard dimensions two through four and the existing sparse and THC
fixtures. Their existing gate, resource, measurement and clear-work
tests exercise the same preparation and walk entry points.

`BlackBoxPrepare` is the named-to-packed binding adapter. The audit checks
that two independent selection registers bind in order inside a packed
selection word, with no declared junk word and no emitted partition gates.
The packed adapter has its own optional normalization representation
and its layout rules remain under Q-0691.
The retained Hubbard fixture checks the two-bit junk word and coefficient
norm 32 through this adapter. Its append execution is covered in the
[packed PREPARE audit](BLACK_BOX_PREPARE_FILE_AUDIT.md).

The source base emits no independent primitive or resource estimate.
Concrete workspace ownership, measurement feedback and amplitudes are
defined and tested by each implementation. External Qualtran
`GateWithRegisters` subclasses and symbolic executable binding remain
open under Q-0693. Native emitters continue to provide concrete append
adapters.
