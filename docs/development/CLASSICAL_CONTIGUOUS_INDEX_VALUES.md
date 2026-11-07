# Classical contiguous index values

Simulation owns the raw numerical callback from unchanged Qualtran v0.7.0
`arithmetic/conversions/contiguous_index.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

`classical_contiguous_index.py` preserves the input objects and evaluates exactly
`nu * (nu + 1) // 2 + mu`, in that order. It does not convert operands to Python
integers, apply an explicit output modulus, or impose unused width metadata.
Native NumPy promotion and overflow remain the original callback's behavior.
The source drawing's minus sign does not replace the executable plus sign.

Compiler owns the existing square/add and direct-column XOR oracles. Their
big-endian output register receives the low-width index XORed into its previous
value and all workspace is cleaned. This is a separate contract from creating
the raw numerical index. Runtime owns measured execution.

The source root declares a Toffoli model but has no full gate decomposition.
Its through-`s` signature also conflicts with the two-argument numerical callback;
the original `call_classically(..., s=...)` raises an unexpected-keyword error.
This slice does not invent an upstream gate sequence or claim that the complete
root execution protocol is implemented by the numerical callback.

Run `python -m examples.algorithms.classical_contiguous_index_values` for the
ordinary `circuit.append(op)` path. It checks all complex input amplitudes with
occupied targets and a spectator, clean work, and restoration by a second XOR.
The emitted one-bit columns oracle has 4 T gates, one measurement, one work bit,
and six declared wires. The original model declares one Toffoli (four T
equivalents), zero measurements and four logical wires; it omits the native
workspace and spectator. Equal T equivalents do not establish gate-flow parity
for a source root with no decomposition.

Reference acceptance covers 101 new cases, including native scalar overflow,
array broadcasting, symbolic operands, exact operator evaluation order and both
existing quantum paths over every small measurement history. Work cleanup uses
the same absolute complex-amplitude tolerance of `2e-12` as the full-state check.
Together with existing contiguous-index and resource-report tests, 158 checks
pass without skips. The 24 paired overflow warnings are retained.
