# Dirichlet advection with native quantum circuits

Open [the walkthrough](walkthrough.ipynb) or run the complete workflow from the repository root.

```bash
python -m examples.synthesis.community.hackathons.qinnovision_2025.advection_equation.second_place.overview.main --figures advection-figures --results advection-results.json
```

The native Circuit retains the original seven-bit, 128-point domain, normalized plateau, thirty-two evolution steps, centered-difference classical reference, four evolution times, and every two-to-seven-bit controlled-rotation comparison. It reproduces the original readout of 32 times the computational-basis probability.

The controlled real rotation uses an exact Gray-code multiplexor. Resource counts come from the explicit U3/CX basis after native conversion. The source paper's resource formula is kept as a separate curve. All ratios are shown on the plot, including values below one.

An independent dense Circuit reference evaluates every complex amplitude at two original nonzero times. The classical baseline retains its original explicit centered update, step size 0.001, and floating-point loop termination. Its measured overshoot is displayed.
