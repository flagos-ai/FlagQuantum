# First-quantization projectile momentum-transfer PREPARE audit

The frozen Qualtran v0.7.0 files are under
`qualtran/bloqs/chemistry/pbc/first_quantization/projectile` at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Inventory | Source file | Git blob |
| --- | --- | --- |
| Q-0375 | `prepare_nu.py` | `682ab3c65449b22c50edc713cbccbf7f3d3f150f` |
| Q-0376 | `prepare_nu_test.py` | `020c80caa5a5c239bfb3d98cd0e82ceed9db917b` |

## Source and native direction

The shared ν signature uses the projectile momentum width for μ
and that width plus one on each of the three ν axes. The m word
and external success flag are through. The projectile μ leaf
produces a right-only `flag`; the forward inequality consumes it
on `flag_mu_prep`. The other two temporary flags reuse the ordinary
ν ownership contracts.

In the locked source, `PrepareMuUnaryEncodedOneHotWithProj.adjoint()`
still declares `flag` on the right. The enclosing adjoint tries to
supply it on the left. Both decomposition and the peak-width query
fail at that edge. The source can still report adjoint gate categories.

The corrected native signature consumes the μ flag on the left.
The native adjoint model uses the original gate categories with these
live widths. Its expanded call order is checked against the reversed
forward graph, with all three temporary edges checked for producer and
consumer compatibility. This is a corrected resource model. The five
declaration-only leaves still lack full physical execution.

## Estimates and measured child

At electron width 6, projectile width 8, and M=524288:

| Scope | Gate accounting | Peak logical qubits |
| --- | --- | --- |
| Locked forward | 3,868 T equivalents | 93 |
| Locked adjoint | 116 T equivalents | Source query fails |
| Corrected native adjoint | 116 T equivalents | 58, native estimate |

The original round-trip resource check divides the combined T
equivalents by four and obtains 996. The expanded native adjoint
adds 19 Clifford gates for the independent uniform m substep at
this size. The declared cost graph omits that child.

The independent three-value m fixture binds the actual uniform
child from the original forward composite and uses its adjoint
for the reverse substep. The native binding uses
`circuit.append(op)`, matches the full complex state for all
32 measurement histories in each direction, and clears all four
work wires. It emits 20 T/T† and five measurements per direction.

The untouched original test file reported seven passes, three
skips and one expected failure. Q-0376 closes the original test
coverage. Q-0375 stays open for the unfixed locked-source adjoint
and the physical projectile μ and related ν leaves.

The subsequent [finite constructor dependency audit](RESOLVER_PROJECTILE_NU_AUDIT.md)
connects original serialized depths 0/1/2 to the source-exact declarations and
six-call forward graph. Its mu inverse declaration retains the original RIGHT
flag, separately from the corrected native model above. The actual uniform-m
child reuses the accepted executable Program; opaque siblings still refuse
physical binding. Constructor registration does not close Q-0375.
