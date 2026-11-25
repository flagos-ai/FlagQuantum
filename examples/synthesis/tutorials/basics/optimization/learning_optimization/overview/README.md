# Bounded integer optimization

Run from the repository root:

```bash
python -m examples.synthesis.tutorials.basics.optimization.learning_optimization.overview.main
python -m examples.synthesis.tutorials.basics.optimization.learning_optimization.overview.comparisons.circuit
```

Open `walkthrough.ipynb` for the complete native qfunc, actual optimization
loop and named sample readout. Minimize `3*x+2*y`, with `x,y` in `0..3` and
`3*x+y>=2`. The native cost is `3*x+2*y+20*max(0,2-3*x-y)^2`. One cost layer
and RX mixing run within four physical qubits and zero workspace.

Each objective call synthesizes and executes the quantum program. The local
Nelder-Mead run has a 32-iteration bound and reports convergence honestly.
The 1000 seeded local samples are drawn from the final executed distribution.
The best observed feasible result is not a convergence or hardware guarantee.

[Workflow mapping](WORKFLOW.md) discloses the encoding and optimizer changes.
The retained eight-bit quantum-program counterpart in the parent directory
preserves the frozen source slack implementation independently.
