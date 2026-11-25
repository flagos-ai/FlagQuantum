# Native quantum programming, part II

Run the complete tutorial from the repository root.

```sh
python -m examples.synthesis.tutorials.basics.quantum_programming.qmod_tutorial_part2.overview.main
```

Open [the walkthrough](walkthrough.ipynb) for all 23 original code stages.
The exercises run reference answers and the later solution cells show their
definitions and measured results. The examples retain the original quantum
register widths, phases, Fourier addition, evolution time and product formula.
The local comparison explicitly uncomputes its temporary register. Controlled
Fourier addition puts the control on the phase block, with the Fourier
transform and inverse surrounding both branches.

The native qfunc exercises use FlagQuantum synthesis. Two exercises use the
public Circuit API. The higher-order helper takes a typed Python operation.
The binding exercise builds one symbolic two-qubit circuit and uses
`bind_parameters` to produce each concrete circuit while keeping the symbolic
template available. This is Circuit binding, without a synthesis runtime
binding requirement. Each sampling exercise uses 1,000 local draws where the
source leaves its shot count unspecified. The Notebook displays native source
and actual resource estimates in place of the hosted viewer.

Independent complex-state checks preserve global phase and verify temporary
bit cleanup. The Trotter reference uses dense Pauli exponentials for the same
product formula. The last expectation is checked against
`sin(alpha) * sin(beta) + 0.5 * cos(beta)`.

[Source mapping](provenance.json) · [Third-party license](../../../../../CLASSIQ_LICENSE.txt)
