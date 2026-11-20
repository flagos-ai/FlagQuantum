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
slot. Its inverse clears the same carries.

`signed_encoding_component.signed_encoding_program` now owns the native sided
Program composition. Forward consumes LEFT ones-complement `x` and returns
RIGHT signed `y`; inverse exchanges the types. Both preserve the physical word
alias. Split/Join retain the type boundary, and existing shared And components
allocate and erase/free each carry. The caller provides n-2 clean wires. The
flat emitter and typed composition share one increment sequence. No foreign
class or protected API is added.

The source class has no decomposition or direct classical implementation.
`on_classical_vals` returns NotImplemented, and `call_classically` raises because
no decomposition is available. These distinct failure layers are tested on the
actual factory. Foreign source capture rejects the LEFT/RIGHT root at its established
through-port boundary. The native sided Program above is independently executable. The source model is n-2 Toffoli and uses its own peak
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


## Native sided Program dependency closure, November 20, 2026

The complete original declaration and test file were revisited with their real
SELECT cost consumer. This delivery closes the native sided composition gap
using the existing register builder, bit views, shared And owner and binder.
The two flat conversion emitters match the published pre-batch instructions
exactly in 496 comparisons over widths 2 through 32, both directions, ordinary
and reversed word layouts, and measurement slots 0 and 9. This comparison applies
no normalization. The shared sequence preserves both existing flat emitters.

Stored and expanded typed Programs match the flat owner's entire ordered stream
at widths 2, 3, 4, 5, 8 and 10 in both directions, including operands and
feedback. Only fresh measurement SSA slots are renamed to the flat owner's
reusable slot. Dynamic flags and conditions are retained. Existing tests compare
the actual unchanged source And and adjoint leaves, including their feedback.
The frozen root has no decomposition, so root gate parity remains unavailable.

Every forward/inverse measurement-history pair at widths 2, 3 and 4 preserves
the full unnormalized complex amplitude, an entangled spectator and zero carry
storage. Nested forward/inverse sided calls execute through normal
`circuit.append(op)` with the flat emitter replaced by a raising stub. Widths
and direction are checked at construction. Negative zero follows the existing
native reversible full-word permutation.

For n bits the measured native primitive account is 4(n-2) T/TDG, 7(n-2)+1 CX,
3(n-2) H, n-2 S, n-2 conditional CZ, n-2 measurements and n-2 resets. Inverse
adds 4(n-2) X. There are n-2 simultaneously live clean carries and n-2 fresh
measurement slots in the typed path. The flat path reuses one measurement slot.
The source root declares n-2 Toffoli independently of these primitive counts.
At n=10 this means 32 T/TDG, 57 CX, eight measurements and eight clean carries,
compared with the separately reported source model of eight Toffoli.

Q-0193/Q-0194 remain open for the recorded foreign capture and missing source
execution boundaries. This delivery adds no parent or capability count.
Evidence uses the `g2-signed-encoding-component-*` receipt prefix.

This batch's concentrated run passed 2938 cases without skips, with 13 existing
upstream or paired-runtime warnings. All 17 local integration checks passed.
The final focused run passed 72 cases. The unchanged original test file again
reported three passes and three skips for its absent decomposition-dependent
modes. All three changed Python files passed strict typing, Black and Ruff.
Integration/Compiler scope, architecture, IR and protected API checks passed.
Three frozen source blobs, Qualtran 0.7.0 and the Kickmix 0.1 Simulator were
verified. Dependency policy retains 687 whole-repository findings and zero
findings in this batch. The prior default-suite docstring failure is reused as
a separately recorded baseline, without a fresh whole-suite pass claim.
