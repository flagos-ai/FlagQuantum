# Native advection methods and resource comparisons

Run [the walkthrough](walkthrough.ipynb) or the complete workflow from the repository root.

```bash
python -m examples.synthesis.community.hackathons.qinnovision_2025.advection_equation.winning.overview.main --figures winning-advection-figures --results winning-advection-results.json
```

The original seven-bit, 128-point domain spans −1 to 4. The workflow builds its centered Dirichlet derivative, all 28 RK4 columns with step 0.038, and ten fresh matrix-exponential steps. The source's index `int(1 / 0.038)` reads time 0.988, and this exact readout is retained.

All three quantum methods retain ten steps and seven physical bits. Native qfunc prepares the 26-point normalized plateau at complex128 precision. Public Circuit composes the original W ordering, custom controlled-RZ sequence, and ordered 127-term Pauli product. Both complete proposed and custom states match independent dense W operators. The Trotter state matches a separate dense ordered Pauli oracle. Global phase corrections are reported alongside the absolute complex-state checks.

The workflow retains the source's three-to-six-bit resource sweep without preparation, followed by its three-bit prepared step sweep `(1, 1, 2, …, 9)`. Static resources are measured after explicit U3/CX conversion. The original input sizes and formulas are preserved.

The observed fidelity against the source RK4 readout is about 24.68% for all three ten-step circuits. The archived proposed plot reports the same low-fidelity behavior. Every curve and error plot shows the actual result.
