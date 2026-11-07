# Classical truncation and duplicate values

This numerical slice follows unchanged Qualtran v0.7.0 source at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Compiler keeps ownership of the
existing sign-copy and sorted-duplicate gate sequences; Runtime owns execution.
The numerical functions add no quantum instructions or workspace.

## Interfaces and ownership

`classical_signed_truncation.py` accepts a value, input/output widths and the
choice of two's or ones' complement. The encoding choice describes both ends;
it does not introduce foreign dtype objects or permit mixed encodings. The
function retains source construction checks, explicit `int(value)`, big-endian
encoding, removed-prefix sign checks and output decoding. Existing integer and
ones' complement codecs remain authoritative. Their shared ones' complement
width guard is reused rather than copied. Bare SignExtend's identity callback
does not require a separate numerical wrapper.

`classical_sorted_duplicates.py` accepts a mapping of classical register values
and returns a new shallow dictionary, matching the source keyword-argument
dictionary. It retains the original `xs` value, uses a temporary `np.asarray`
view for sortedness and adjacent equality, and applies the original augmented
XOR to `flag`. An ndarray flag can therefore change in place even though the
mapping itself is copied. Extra fields remain present and unmodified. Shape,
width and controlled-mode metadata unused by the source callback do not become
new validation arguments.

Input-prefix qubits cleared by quantum truncation are released logical inputs,
not newly allocated clean work. Duplicate-detection difference and predicate
records belong to its existing compiler workspace. Classical validation is not
proof of arbitrary dirty-output quantum erasure. Ones' complement negative-zero
bit patterns must remain distinct in coherent quantum tests even when their
decoded numerical value is zero.

## Running the user path

```sh
python -m examples.algorithms.classical_truncation_and_duplicates
pytest -q tests/unit/test_classical_signed_truncation.py tests/unit/test_classical_sorted_duplicates.py
```

The example uses `circuit.append(op)` and existing compiler operations. Coherent
sign truncation clears the redundant two-bit prefix and preserves both raw zero
encodings plus a spectator; applying the same fanout restores the original state.
The prefix remains declared in the native IR and is eligible for reuse. This is
not a claim of immediate statevector memory reclamation or extra allocated work.

Numerical truncation first checks the mapped metadata and only then calls
`int(value)`. Slice/index widths are not converted to Python integers. It rejects
a nonredundant prefix before invoking the narrower decoder. Tests cover 18
width-pair configurations, both encodings, malformed/NumPy/symbolic inputs and
codec substitution without editing the reference implementation.

The duplicate example uses the existing operand-preserving route on a two-word
uncontrolled input. All valid sorted inputs and both flag values are in a complex
superposition. Tests separately exercise the original source-alias route and
its output bindings, active controlled cases, every small measurement history,
spectators and clean workspace. Input mappings are copied shallowly: if `xs` and
`flag` refer to the same ndarray, toggling the flag can change `xs` after the
sortedness assertion. The source behavior is preserved, without adding such
aliasing to the compiler's accepted disjoint wire layouts.

## Resource bases

| Consumer | Emitted primitive instructions | Pinned analytical model |
| --- | --- | --- |
| Four-to-two-bit truncation | 3 CX, 0 T, 0 measurements | 1 logical Clifford component, 0 T, 0 measurements |
| Two one-bit-word duplicate detection | 4 T/T-dagger, 1 measurement, 5 reserved work, 8 declared wires | 4 T equivalents, 1 measurement, 7 peak logical qubits |

Both original and native truncation primitive streams contain the same three
CNOT instructions. The default source QEC classifier counts the entire fanout
as one Clifford component. No primitive count is changed to fit that abstract
category. Duplicate workspace is explicitly reserved by the existing emitter;
the source lifetime model is reported separately from declared circuit capacity.

## Source interfaces that remain different

Original `HasDuplicates.on_classical_vals` ignores `ctrl`, including in controlled
mode. For sorted `xs=[1,1],flag=0,ctrl=0`, the original callback and native value
callback return `flag=1`, while the unchanged original decomposition and native
gate path return `flag=0`. This is reproduced directly in the test suite and
remains an open source-interface issue. The port does not repair the callback or
the source to pretend that these interfaces agree.

Existing source alias/erasure issues for longer lists and uncontrolled singleton
decomposition remain recorded in the integration ledger. The existing explicit
operand-preserving compiler route and original-alias diagnostic route keep their
separate scopes; this numerical slice changes neither gate implementation.
Unused length/dtype metadata, missing dictionary fields and multidimensional
array diagnostics passing a raw callback do not certify valid quantum inputs.

Acceptance: 836 truncation and 504 duplicate cases are new. Concentrated regression
passes 2,757 cases without skips, retaining 67 reference/native numerical warnings.
Four production/example files pass changed-surface strict typing, formatting,
ownership and architecture checks. Five pinned source blobs for the callbacks,
value codecs and shared gate dependencies remain unchanged. Core APIs and
serialized schemas are unchanged.
