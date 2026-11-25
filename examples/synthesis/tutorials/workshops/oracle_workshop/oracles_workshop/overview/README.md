# Native Oracle Workshop

Open [walkthrough.ipynb](walkthrough.ipynb) from the repository root, or run:

```bash
python -m examples.synthesis.tutorials.workshops.oracle_workshop.oracles_workshop.overview.main
```

The complete Notebook preserves all nineteen source code stages, including labeled
exercise reference solutions and full final solution replays. It keeps the original
one/two-bit x/y registers for2*x+y==4, three-bit balanced Deutsch-Jozsa, and two/two-bit
x/y registers with two Grover rounds for x-y==2.

Both phase-kickback lessons execute: an explicit root-owned auxiliary prepared and
reversed through within/apply, and a direct conditional phase. Eight complete native
program states are compared against independent Circuit references. Three complete
eight-row complex phase tables preserve the original magnitude/angle-in-pi readout.
Four batches use actual1000 local draws; source implicit sampling defaults become
explicit and reproducible.

The original fixed qperm,phase_oracle,grover_search and struct applications expand
into native qfunc helpers and named ports. This demonstrates the specific algorithms;
it does not claim a generic public callable/qperm/QStruct API. The root owns a clean
auxiliary that helpers borrow and return to zero. `drop` ends its lifetime after
uncomputation; it does not reset or erase entanglement.

Public synthesis resource reports replace the provider program viewer. Counts are
before Runtime compilation and are not hardware-basis counts. The frozen sibling
quantum counterpart and all216 native quantum programs remain unchanged. No Classiq
or Qmod source is imported. No hardware,scalability,speedup or resource-optimality
claim is made. See [WORKFLOW.md](WORKFLOW.md) and [provenance.json](provenance.json).
