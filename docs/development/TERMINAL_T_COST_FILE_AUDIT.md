# Historical terminal T estimate source-file audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Checklist | Frozen source | FlagQuantum evidence |
| --- | --- | --- |
| Q-0816 | `resource_counting/t_counts_from_sigma.py` | `estimate_terminal_t_cost` counts each explicit T or T-adjoint as 1, Toffoli as 4, bit Fredkin as 7 and forward binary AND as 4. Admitted rotations use the shared precision rule rounded before multiplying the terminal count. |
| Q-0817 | `resource_counting/t_counts_from_sigma_test.py` | The original symbolic-precision fixture runs unchanged. `test_terminal_t_cost.py` compares the same rotation groups, symbolic multiplicity, numeric bindings, controlled stabilizers and explicit gate counts. `test_terminal_t_cost_file.py` checks both frozen hashes, rounding order, model separation and undecomposed arithmetic. |

Controlled S and T require a precision that the frozen source does not
provide. The source asserts and the native estimate raises a descriptive
`ValueError`. Controlled rotations with symbolic angle have no stabilizer
verdict and raise `NotImplementedError` on both sides. The internal native
helper has no deprecated alias. For undecomposed native arithmetic, the
direct estimator requires expansion or a declared recursive cost. The frozen
function silently contributes zero for the corresponding composite key.

The exact Fredkin stream contains seven T/T-adjoint primitives, matching the
historical terminal estimate. The QEC model assigns one controlled swap,
equivalent to four T under its default conversion. Three rotations priced at
precision 0.0001 contribute 75 historical model T, using 25 per rotation.
That precision-sensitive estimate does not synthesize the physical sequence.
Existing Fredkin tests cover gate operands, coherent amplitudes and both
measurement outcomes of the controlled extension.
