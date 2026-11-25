# Original Ising walk and signed Pauli dependency closure

Reference: unchanged Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, Google LLC, Apache-2.0.
The existing source fixture checks Pauli SELECT source/test/notebook blobs
`41e2f8ddb341fbbdd2d004ef533041bd24a5898c`,
`81ccdd543c60783cecee6a87ae238b050ed3b28b` and
`6513d5faf7b227cd26534f30e05a7d3b55a29f5a`.
Reference tests require the actual Kickmix 0.1 Simulator.

## Complete dependency ownership

Compiler lookup owns a typed bounded Pauli SELECT graph. Selection is a
scalar MSB-first bounded uint, target is an opaque THRU word, and optional
control is a scalar QBit. Existing unary traversal owns And computation and
measured erasure. Each controlled signed Pauli leaf keeps its original call
boundary and only nonidentity target positions. One position is a scalar
QBit; multiple positions form a QBit array. Empty positive identity retains
its control-only leaf. Generic inverse reverses the actual primitive leaf
stream and source unary calls, retaining explicit compute/erase ownership.

Ecosystem converts actual DensePauliString objects immediately to native
signed words. This bypasses the original failing object-array serialization
without patching it or retaining SDK objects. The source takes the sign of
the real coefficient; the adapter follows that convention. A zero or
nonfinite real part has no accepted unitary native leaf. Zero-width selection,
zero-width target and uncontrolled singleton traversal remain explicit
unsupported boundaries rather than fabricated source decompositions.

The alias capture helper is now shared by default packed LCU PREPARE and
actual Pauli walks. The previous default-LCU codec/resolver implementation
is replaced, avoiding a second alias implementation. Its existing exact
directional constructor and original codec refusals remain authoritative.

Pauli walk capture binds the actual SELECT block and actual alias reflection,
ordinary and source-positive controls, and explicitly retained signal words.
The source Pauli selector declares `adjoint() == self`; block inversion keeps
that child identity, while the standalone generic Pauli inverse remains a
reversed stream. All numerical execution stays outside Compiler.

## Numerical and source evidence

Signed X/Y/Z and identity tables cover ordinary and both control polarities,
both generic directions, padded addresses, comparable calls/operands, complete
complex logical operators, every small raw measurement history and clean
private work. Primitive inventories also match the unchanged source forward
decomposition; inverse inventory comparison uses the same self-adjoint source
operator, not an invented direct Cirq decomposition for generic Adjoint.
The original Pauli constructor example is captured and checked independently.

The actual four-site Ising `_walk_op` example has eight signed Pauli terms,
three selection bits, four target bits, and alias precision mu=4. Its twelve
signal bits are explicitly retained, giving nineteen logical data bits.
Both ordinary and source-positive walks compare complete sparse complex
outputs on superpositions involving selection, target and control inputs.
Zero and one raw histories plus all-history native erasure certificates check
private scratch. Exact inverse restores the complete joint input, including
the retained signal environment. Binding and `circuit.append(op)` are tested
with both external SDK modules unavailable after capture.

This is not a dense operator over every possible nineteen-bit environment or
enumeration of every large-example history. The source physical root is forced
and its real signal allocations are observed before release; selection-only
projected tensors are not used as a false unitary reference.

## Actual instructions and models

Eight directional root/selector inventories are recorded in
`g2-ising-pauli-resources-20261125.json`. The ordinary actual Ising walk emits
178 T-family, 424 CX, 128 H, 34 measurements and resets, 43 CZ, 84 X,
34 S, four Z and two carrier phase instructions. Private scratch peak is
eleven, retained logical data is nineteen, and total peak is thirty.
The original model separately reports six CSwap, 34 And, 271 Clifford,
34 measurements and peak thirty. Neither CSwap/And models nor scalar-carrier
recipes are substituted for emitted primitive counts. Released leases and
numerical zero-state proofs remain separate checks.

The batch regression has 162 passes; final expanded focused scope has eighteen
passes, including one added original-constructor case. Strict typing covers
seven files. Existing local integration and source receipts are reused where
their execution paths did not change; new integration is recorded separately.

## Remaining Walk file scope

The original THC and sparse chemistry constructor dependencies remain open.
The original source ABI still omits retained signal ports; native direct
composition supports them and the old ABI refuses lossy binding. These source
boundaries and large chemistry work prevent marking the complete Walk
constructor closed or increasing the 117/303 constructor inventory count.
G2 continues within the same integration PR; this does not enter G3 or merge it.
