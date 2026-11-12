# Grover arithmetic oracle

This example expresses a seven-qubit arithmetic marking predicate directly as a
native FlagQuantum qfunc, then composes it with an exact uniform-state reflection.

```sh
python -m examples.synthesis.tutorials.workshops.grover_workshop.grover_workshop.main
python -m examples.synthesis.tutorials.workshops.grover_workshop.grover_workshop.comparisons.circuit
```

The second command compares the complete complex amplitudes and register order
against an independently constructed dense Circuit. The native oracle uses no
kickback ancilla or other workspace. The related workshop Notebook is tracked
separately and remains pending.

Reference identity: `tutorials/workshops/grover_workshop/grover_workshop.qmod`
at `2bfd835e73b6882fcc515a725098b04766c238e7`.

Original source and importer regressions remain in test fixtures. The shared
third-party license applies: see [the license](../../../../CLASSIQ_LICENSE.txt).
