# Complete source workflow acceptance

Retain the six executable source cells: seed 1, four normalized real random
amplitudes, equal-weight XX/XZ/ZX/ZZ encoding, zero/one repetition programs,
actual execution, filtered good-block tables and fraction calculations.
Do not replace either quantum experiment with a classical analytic answer.
Reuse the existing native qfunc encoding, adjoint and within/apply reflection.
No production API or dependency is added.

Correct the source final all-row normalization assertion explicitly. Read the
actual block-zero marginal separately and require probability 1/4 before and
1 after amplification. Check normalized total probability independently.
Compare complete states without phase alignment using independent dense Pauli
SELECT and block-only reflection Circuits, then check projected amplitudes
against the independently constructed H tensor H. Keep data and selector order,
nonzero data models and hidden-work-zero coverage. Existing quantum-program
acceptance retains nonzero selector columns and budget/repetition refusals.

Run all five Notebook code cells, both CLI launchers and source provenance
checks. This source has no executable plots; explanatory cloud images and
viewer calls become local prose and measured resource reports. Exact CPU
probabilities replace sampling. No hardware, finite-shot, robust nonunitary or
general optimization claim is made.

```sh
python -m pytest tests/integration/test_synthesis_oaa_notebook.py tests/integration/test_synthesis_native_oaa.py -q
```
