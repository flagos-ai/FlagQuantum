# Whitebox fuzzing with quantum search

Run the native workflow from the repository root:

```bash
python -m examples.synthesis.applications.cybersecurity.whitebox_fuzzing.overview.main
```

[walkthrough.ipynb](walkthrough.ipynb) preserves all eleven source code cells:
the branching function, seeded classical samples, predicate oracle, original
iteration estimate, ten-round search, 500 local observations, labeled rows and
two branch-count charts. The parent quantum-only example is unchanged.

Users express the target branch as a predicate and compose its phase marker
with Grover diffusion. The existing native helpers in the parent directory
perform that work; no physical wire numbering or bitstring decoding is needed.
The complete workflow uses two six-bit named registers and the source width
ceiling of 25. Named probabilities feed actual 500-observation local draws.

Classical draws preserve the source NumPy RandomState seed 3 and half-open range
`[0,63)`. A documented seed 1913 makes the previously unspecified quantum draws
reproducible. The exact branch predicate has six marked pairs. The source
heuristic assumes two and prints 36 iterations; the actual source circuit uses
ten. Both values are retained and explicitly qualified. This demonstration does
not claim an optimal iteration choice, hardware calibration or quantum speedup.

Independent Circuit checks compare every complex amplitude at zero, one, two
and ten Grover rounds, including absolute phase and clean auxiliary storage.
The bare oracle is checked against a separate classical truth table after
Circuit preparation. Full classification is exhausted over all 4096 inputs.
Plots retain all five branch labels, including zero-count outcomes, and weight
observations by their actual counts.

See [WORKFLOW.md](WORKFLOW.md) and [provenance.json](provenance.json) for source
mapping and acceptance scope. Full Notebook status is tracked separately from
the already covered quantum counterpart.
