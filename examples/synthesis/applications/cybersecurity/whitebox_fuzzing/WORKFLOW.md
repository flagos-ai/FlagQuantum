# Native coherent search

Run the native program from the repository root:

```bash
python -m examples.synthesis.applications.cybersecurity.whitebox_fuzzing.main
```

`main.py` declares logical numeric outputs, prepares the uniform state and
composes the problem oracle with a phase-exact uniform reflection. Helpers
contain only logical predicates and transformations. Users do not assign work
qubits, design control ladders or decode physical bit strings.

The complete original problem is retained: two six-bit integers, all4096 assignments and ten Grover iterations. The marked condition is x+y<9 and (x modulo4)*(y modulo2)=1; exactly six assignments satisfy it.

The native sparse predicate path bounds the input domain to twelve bits and
at most32 satisfying minterms, with all-true/all-false cases explicit. Existing
polynomial budgets remain intact. Clean factors compute each minterm, apply its
phase and uncompute. Arbitrary finite phase coefficients remain coherent; this
is a bounded exact synthesis path, not a general optimal Boolean minimizer.

`comparisons/circuit.py` independently enumerates the mathematical predicate
and builds basis signs with explicit Circuit X/H/CCX ladders. Its extra scratch
is reference-only and checked zero; native work allocation is independent.
Tests compare full complex states without phase alignment, including the
unchanged source after logical register mapping, and enforce actual budgets.

Classical application analysis and complete Notebook workflows are tracked
separately. This acceptance is local CPU quantum-program correctness only.
