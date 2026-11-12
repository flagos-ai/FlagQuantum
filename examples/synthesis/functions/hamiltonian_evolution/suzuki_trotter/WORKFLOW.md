# Native Suzuki-Trotter construction

The source Notebook has one code cell: define the two-term Pauli Hamiltonian,
allocate three target bits and synthesize its first-order single-step evolution
with classical time `x`. The native program uses `fq.Observable`, `@qfunc` and
`suzuki_trotter` with the same terms, coefficients and declared order.

FlagQuantum specializes a supplied finite real time at synthesis, rather than
retaining an unbound symbolic CReal in a cloud artifact. `build_program(0.37)`
and `build_program(-0.29)` complete the local construction workflow. The example
also executes and reads named probabilities; independent Circuit checks compare
complete complex amplitudes, absolute phase and bit order. No cloud service or
symbolic-parameter reuse is claimed.

Observable index zero means target element zero, the least significant bit.
The independent Circuit uses explicit Kronecker products with reversed array
order. Tests cover first/second order, multiple repetitions, coherent inputs,
identity global phase, a target offset by another register, mutable helpers,
zero workspace, invalid data and resource limits. Product formulas approximate
a noncommuting Hamiltonian sum; their numerical product is validated without
claiming an unmeasured approximation bound.

Run the native program and independent Circuit comparison:

```sh
python -m examples.synthesis.functions.hamiltonian_evolution.suzuki_trotter.main
python -m examples.synthesis.functions.hamiltonian_evolution.suzuki_trotter.comparisons.circuit
```
