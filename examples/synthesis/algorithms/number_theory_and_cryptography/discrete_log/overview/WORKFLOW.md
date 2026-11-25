# Complete Discrete Log workflow

| Original code cells | Native workflow |
| --- | --- |
| 5,7,9 | Native imports, modular-power helper and inverse-QFT algorithm |
| 14,15,16,18,20 | Mod-5 synthesis, resources, 4000 shots, inverse filter and logarithm 3 |
| 25,26,27 | Mod-13 five-bit fractional phases, four-bit modular target and synthesis |
| 30,32,33 | Rounding, coprime filtering, modular inversion and leading answer |
| 35 | Nonempty candidate list and original leading-ten reconstruction assertion |
| 37,38 | Original t=5 and t=8 real-log-ratio illustrations |

Native qfuncs allocate registers, prepare uniform phases and the target value one,
apply reversible modular exponentiation, and invert QFT. Mod-13 reuses the existing
native example. Original width limits 13/23 and shot counts 4000/10000 are retained.
Physical resources are 8/16 qubits, 1/2 clean working qubits, 65/1068 operations and
depths 50/872. These are static instruction counts, not a hardware decomposition.

Readout uses named integer distributions for mod-5 and exact Fraction labels in
multiples of 1/32 for mod-13. Users do not decode bitstrings. Seeds 1605/1613 make
local sampling reproducible; the hosted source used unseeded execution. Independent
Circuit probabilities sampled with identical seeds must yield identical draws.

Mod-5 preserves beta in {1,3}, beta times inverse(alpha) modulo four, including
the source omission of a final modulo, and checks every selected answer is 3.
Mod-13 preserves Python round-to-even, coprime rounded beta, modulo-twelve recovery,
and the source assertion on the leading ten retained rows. Lower-count incorrect
candidates remain in the output; approximate phase estimation is not presented as
an exact deterministic solver. The leading recovered logarithm is 8.

Both source plots use the real ratio log(3)/log(7), approximately 0.564575, rather
than the modular logarithm 8. They are labeled illustrative and never presented as
sampled marginals. The original 32/256-point curves and normalization are retained.

Independent whole-power permutation Circuits and a separate modular-function FFT
check every complex amplitude at absolute tolerance 2e-11, without global-phase
alignment. Dirty-work probability must be below 1e-24. Four additional nonzero
base bindings check ordering and composition. Invalid bases, labels, counts and
sampling bounds are rejected; existing modular-power tests cover resource budgets.

```sh
python -m pytest tests/integration/test_synthesis_discrete_log_notebook.py tests/integration/test_synthesis_native_discrete_log.py tests/integration/test_synthesis_native_modular_exp.py -q
```
