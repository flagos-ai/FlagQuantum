# Signed conversion declaration whole-file audit

Q-0193 and Q-0194 cover the complete
`arithmetic/conversions/ones_complement_to_twos_complement.py` and its original
test file at Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`. Both frozen Git blobs are checked.
The source defines one immutable class with bitsize, cached signature and direct
cost graph, one ten-bit factory and its documentation specification.

## Contract and ownership

The original input is LEFT `x` with QIntOnesComp and output is RIGHT `y` with
QInt. This is an actual one's-complement encoding contract. The comment about
controlled complement followed by increment describes the separate
sign-magnitude construction. For example one's-complement -1 is 1110, and the
native one's-complement increment maps it to 1111. Applying the comment's extra
complement would instead produce 1010. REGISTER_SIGNED_ENCODING.md keeps this
source distinction and the duplicate negative-zero boundary explicit.

Compiler `registers/bits/signed_encoding.py` owns the concrete
`ones_complement_to_twos_ir` and separately named `sign_magnitude_to_twos_ir`.
Simulation `ones_complement_encoding.py` owns numeric encoding. Core owns the
existing dtype and sided register metadata. The physical native conversion
binds one in-place MSB-first word, n-2 clean carries and a reusable measurement
slot. Its inverse clears the same carries. No new foreign class or API is added.

The source class has no decomposition or direct classical implementation.
`on_classical_vals` returns NotImplemented, and `call_classically` raises because
no decomposition is available. These distinct failure layers are tested on the
actual factory. Native capture rejects the LEFT/RIGHT root at its established
through-port boundary. The source model is n-2 Toffoli and uses its own peak
fallback. It cannot establish an original root-level gate stream.

## Direct dependencies and actual consumers

Existing signed-encoding tests compare the original QIntOnesComp numeric domain,
every admitted encoding, inverse, complex spectator state and all small raw
measurement histories. Shared And and adjoint leaves are compared gate by gate,
including control polarity and classical feedback. AND, controlled XOR and
scalar encodings stay in their existing owners.

The source declaration is a real dependency of first-quantized potential SELECT.
`first_quantization_resource_model._signed_conversion_cost` feeds two edges,
each repeated six times, at widths p and p+2. The audit checks both edges against
the original conversion's full category and peak model across five sizes.
A replacement test swaps only that existing child and verifies that both
unchanged consumer edges use it. The existing full SELECT model comparisons
are rerun with the batch. No production model registry is introduced.

## Complete original tests and resource separation

The original test file contains two functions. Its factory autotester runs
construction and model checks and skips decomposition, count comparison and
dtype consistency modes that require the missing decomposition. Its five-bit
BloqBuilder test checks three Toffoli in the complete child graph. The untouched
file reports three passes and three skips, recorded separately from native
execution evidence.

At width ten the native increment emits 32 T/TDG and eight measurements using
eight carry wires. The independent source declares eight Toffoli. Existing
five-bit evidence records twelve T/TDG, 22 CX, three measurements and three clean
carries. These are separate measurement-assisted primitive and declaration
accounts. The sign-magnitude path adds its independent controlled complement.

Q-0193 and Q-0194 retain the missing source decomposition, unsupported source
classical call, generic sided physical capture and original skipped modes.
The native full-word extension maps duplicate negative zero to the minimum
two's-complement word to preserve reversibility. This is documented native
behavior beyond the source's declared numeric domain. The batch does not claim
an original executable conversion where the frozen source provides none.
Logs are `ftqc-reference-env/g2-signed-conversion-file-*.log`.

The concentrated regression passed 106 cases without warnings or skips. The
new test passed strict typing, formatting and lint. Compiler scope, architecture,
multilevel IR and protected API checks passed. Existing full-word complex,
negative-zero, small-branch and shared-leaf evidence is reused. The source
factory, SELECT child-resource edges and replacement are additional integration
evidence, without altering the native arithmetic implementation.
