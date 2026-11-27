# Learn high-level quantum programming with FlagQuantum

Run these editable Notebooks from the repository root in the project Python
and Jupyter environment. They use local CPU execution and need no cloud account.
Each lesson shows its central program directly, explains the quantum semantics,
and offers one modification to try. Runtime outputs are produced when you run
cells; committed files contain no precomputed results.

| Lesson | Learn | What FlagQuantum supplies |
| --- | --- | --- |
| [1. States](01_states.ipynb) | Prepare a Bell state and read named outcomes | State-preparation implementation and execution |
| [2. Computation](02_computation.ipynb) | Compute x*x+y coherently | Arithmetic, result width and temporary cleanup |
| [3. Predicate phase](03_predicate_phase.ipynb) | Observe a conditional phase through interference | Predicate construction and controlled phase |
| [4. Uncomputation](04_uncomputation.ipynb) | Use a computed value without retaining it | Inverse arithmetic and temporary lifetime |
| [5. Constraints](05_constraints.ipynb) | Compile one program for two connectivities | Gate decomposition and routing within bounds |
| [6. Variational application](06_variational.ipynb) | Close a classical/quantum optimization loop | Bound program execution and local optimizer |

These are FlagQuantum-authored lessons, not additional upstream migration cases.
They do not increase the 212-workflow acceptance denominator. Existing full-scale
application reproductions remain in the five category catalogs.

Independent acceptance is in
`tests/integration/test_synthesis_learning_path.py`. The checks cover complete
complex states, clean temporary storage, named outputs, connectivity and actual
variational energy evaluations.

The variational lesson also contrasts concrete qfunc specialization with the
existing Circuit runtime-input route. Three nonzero tensor angles reuse one
accepted plan and check actual energies and PyTorch gradients. This does not
claim symbolic runtime inputs for qfunc or a compile cache.

To execute all lessons and their references:

```sh
python -m pytest tests/integration/test_synthesis_learning_path.py -q
```

The state lesson also requests actual finite-shot counts through existing
`fq.counts`, selecting the synthesized Circuit's named `pair` mapping. This
creates a separate measurement plan; sealed synthesis Programs currently refuse
shot/output overrides. Exact named probabilities, sampled frequencies and
reported shot uncertainty are kept distinct. No second sampling API is added.
