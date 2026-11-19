# Native fresh option pricing workshop

```bash
python -m examples.synthesis.tutorials.workshops.finance_workshops.option_pricing_workshop.overview.main
```

`walkthrough.ipynb` retains all 12 source stages and the original
five-bit problem, financial parameters, width budget 28, epsilon0.05,
alpha0.01 and final tolerance factor10. Fresh classical data feeds the
existing native qfunc option preparation and Grover reflection. No rounded
quantum-counterpart constants or source importer are used.

Shared implementation lives in the application option-pricing `overview`:
`model.py` computes the fresh 32-point distribution and payoff rotations;
`reference.py` builds independent Circuit and direct-vector reflections;
`workflow.py` performs bounded actual sampling and count-only estimation;
`plots.py` renders the original distribution/payoff diagram.

The iterative estimator is the already validated finite local profile from
native quantum counting, not a claim of proprietary Classiq IQAE equivalence.
It uses simultaneous Hoeffding intervals, at most64adaptive steps, powers0..16
and256fresh binomial samples per batch. Each newly requested structural power
compiles and executes once; cached exact local marginals supply fresh draws.
No runtime parameter binding or hardware timing claim is made. The estimator
receives counts only; exact expected payoff is an independent validation value.

Full six-qubit checks retain the dropped asset as private environment, not clean
compiler workspace. Seeds and all counts are recorded. This local educational
example makes no cloud, hardware, pricing advantage or scalability claim.
