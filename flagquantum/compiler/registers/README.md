# Register decompositions

Compiler owns shared register-level instruction decompositions here.
`arithmetic.py` and `_arithmetic_kernels.py` implement arithmetic circuits,
`_fourier_kernels.py` and `phase_gradient.py` implement Fourier and phase
primitives, and `qrom.py`, `qroam.py` and `unary_iteration.py` implement lookup
circuits. Construction and algorithm consumers compose these instructions.

Dependencies are the standard library, Core CircuitIR and Instruction, and
sibling register primitives. This directory must not capture Python syntax,
choose global construction schedules, execute statevectors or submit jobs.
Preserve bit order, clean-workspace preconditions and dynamic measurement
semantics when changing decompositions.

The user workflow enters through `flagquantum.synth.qfunc` and `synthesize`.
Register helpers are internal implementation entries. Start with the owning
kernel and extend an arithmetic or cleanup scenario before changing callers.

```bash
python -m examples.synthesis.tutorials.arithmetic.arithmetic_implementations.main
python -m pytest -q tests/team/compiler/test_construction_arithmetic_implementations.py tests/integration/test_synthesis_arithmetic_implementations.py tests/integration/test_synthesis_register_cleanup.py
```

The example prepares all two-bit inputs and computes `x*x+y`. It verifies all
16 output tuples with probability 1/16 through the public CPU workflow. Cleanup
tests additionally check that temporary register storage returns to zero.
