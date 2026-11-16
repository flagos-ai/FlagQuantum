# Gaussian preparation through QSVT and exact amplification

Run from the repository root:

```sh
python -m examples.synthesis.community.paper_implementation_project.quantum_state_preparation_without_coherent_arithmetic.stateprep_guassian_using_qsvt.main
python -m examples.synthesis.community.paper_implementation_project.quantum_state_preparation_without_coherent_arithmetic.stateprep_guassian_using_qsvt.comparisons.circuit
```

The native program retains eight fractional data bits (raw word/256), the full
29/28 phase sequences, selected inverse sine block and two exact amplification
rounds. Both visible attenuation and reflection-marker ports must finish clean;
compiler work bits are verified separately. This is the quantum algorithm, not
a precomputed Gaussian solution vector.

The frozen original incorrectly declares success amplitude 0.39579583583036.
It remains rejected. Successful acceptance uses the existing approved revision's
0.39191102421483637, with all phases and unitary effects unchanged. The native
builder rejects the incorrect original value instead of silently replacing it.

The native implementation combines branch phase polynomials and merges adjacent
basis changes around commuting CRX gates to fit existing effect limits. A
selected inverse uses an equivalent phase polynomial between X-axis rotations.
No helper, construction, control or ownership limit is raised.

The independent Circuit uses explicit controlled-Y/projector matrices and two
six-bit zero-group CCX computations with restored shared scratch for reflection.
It compares all complex amplitudes, including absolute phase, against native
and approved source states. Further tests cover prepared overlap, complete model
identity, complex coherent inverse branches, restoration, alias rejection,
invalid declared overlaps and resource budgets.

Native resources are 18 physical bits, including 5 compiler work bits and two
explicit clean ports, with 3,757 operations and depth 3,186. The approved source
uses 17 physical bits, 3,194 operations and depth 2,873. Original refusal and
approved resource records are preserved. This does not certify the full Notebook
workflow or claim performance superiority.
