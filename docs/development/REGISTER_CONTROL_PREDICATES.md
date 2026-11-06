# Structured control predicates

A control group contains row-major activation values. Each word is encoded
big endian. Raw unsigned, bounded and Montgomery words use the same bit
layout. The caller supplies stored Montgomery words and any external bounds.
Signed controls support two's complement and ones' complement.

A predicate preserves the input words and writes its final truth value and
prefix conjunctions into clean caller-owned qubits. These prefix values
remain live until the matching measured uncompute. Two controls use one
And. For n controls, the forward path has 4(n-1) T/T† and the reverse path
has n-1 measurements. The n-2 prefix outputs and one target are included
in the total width of 2n-1 qubits.

The component wrapper invokes a child with a positive control. A single
negative control adds two X gates; a multi-bit condition uses the predicate
compute and uncompute. Child work and classical slots are separate. The child
preserves the predicate and the wrapper's other registers.

The tests read pinned source gates and resource models for shaped controls,
controlled X/Z, and an addition with its own measured cleanup. The addition
example has 12 T/T†, three measurements, and three work qubits across the
two levels. Every small measurement history retains the full expected
complex amplitudes and restores the work qubits.

- [CtrlSpecAnd](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/ctrl_spec_and.py)
- [ControlledViaAnd](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/controlled_via_and.py)
