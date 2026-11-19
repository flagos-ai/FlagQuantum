# Hidden shift from high-level Boolean functions

Run all three original-size scenarios locally:

```sh
python -m examples.synthesis.algorithms.number_theory_and_cryptography.hidden_shift.overview.main
```

Open `walkthrough.ipynb` for the complete teaching workflow. The simple four-bit
case returns `[1,0,0,0]`. The original 16-bit bent function and its dual recover
bits `[1,3,9]`. The no-dual experiment uses an indicator, 2048 actual observations
and exact binary linear algebra to recover the same shift.

```python
from examples.synthesis.algorithms.number_theory_and_cryptography.hidden_shift.overview.workflow import run_estimate

result = run_estimate("no_dual", seed=1502)
print(result.recovered)
print(result.resources)
print(result.maximum_state_error)
```

User functions express Boolean updates such as `ind ^= s[i] & s[8+permutation[i]]`.
Compiler selects the reversible gates and clean workspace. All absolute complex
amplitudes and same-seed samples are checked against an independent Circuit.
The no-dual route has 17 logical bits and four clean work bits, 21 physical bits
in total. Named readout explicitly allows 17 bits; the normal 16-bit guard is
preserved.

SymPy is an optional example dependency for exact GF(2) arithmetic. It replaces
the source Notebook's `galois` package without introducing a production
dependency. No remote account or source importer is used.

See [workflow acceptance](WORKFLOW.md), [source identity](provenance.json) and
[third-party license](../../../../CLASSIQ_LICENSE.txt).
