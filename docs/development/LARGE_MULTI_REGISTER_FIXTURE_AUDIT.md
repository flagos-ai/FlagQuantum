# Large and shaped register fixtures

Q-0495 through Q-0497 match the inventory blobs at Qualtran v0.7.0
commit `bcb17bb25360ed68ce3bc084645db3e929799389`. The original
many-register test passed unchanged. Native fixture code lives under
tests and uses the existing primitive and typed Program owners.

## Repeated AND ladder

The literal first control value of each source AND is applied to the
incoming prefix. The new selection bit is positive. The inverse ladder
uses the same control values in reverse order. The source graph reader
compares each physical operation, operand and classical feedback slot.
The controlled atomic leaf is identified as CX by its complete source
tensor. This supplies an explicit native lowering for the fixture.

At two and three selection bits, every measurement history was projected
directly on a coherent input over all logical basis states. The larger
four-selection-bit and five-repetition fixture uses endpoint histories
and an AND channel certificate over the full logical basis. The
certificate checks the necessary work state at every AND erasure and
establishes the same map for all 32,768 histories. The complete vector
agrees with the untouched source tensor, with the unnormalized branch
factor and zero work. The zero-repetition fixture allocates no work.

The large fixture emits 60 T-family instructions, five atomic action
CX gates within the complete ladder stream, and 15 measurements across
eight wires. Its source qubit estimate is eight. The full source QEC
query fails at the opaque `C[TestAtom]` leaf, which cannot decompose.
The executable gate count and the tensor evidence leave this source
resource boundary open in Q-0495.

## Shaped words and typed companion signatures

The shaped fixture applies X, then four two-bit actions in row-major
order. Each pair has its own split and join. The three-bit spectator
also splits and rejoins without a physical operation. The direct
source calls, nonconsecutive physical port map, tensor axes and full
4,096-entry complex output match. The physical stream is one X and
four CX, with zero work or measurements and 12 wires. The upstream
`TestTwoBitOp` supplies a tensor but no QEC decomposition.

The three companion signatures retain the unsigned bound, fractional
width and signedness. Two source companions declare no decomposition.
The composite companion fails when wiring an unsigned `d` word into a
fixed-point input. The native executable binder separately rejects the
bounded unsigned and fixed-point descriptors. Q-0496 stays open for
these execution boundaries. Q-0497 is covered by the signature,
binding, action and tensor comparisons.

The concentrated native suite passed 209 tests. Strict typing,
formatting, team scope, architecture, IR and public API checks passed.
