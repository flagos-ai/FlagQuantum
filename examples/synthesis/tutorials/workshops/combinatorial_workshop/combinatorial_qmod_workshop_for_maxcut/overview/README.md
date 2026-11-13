# Native MaxCut Workshop

Complete original eighteen-cell MaxCut Workshop: five vertices, six edges, four QAOA layers, all eight angles, original 0..1-radian schedule, forty actual objective executions, actual 1,000 draws, observed maximum cut and original/solution graph views.

Use the unchanged native quantum function and its composable RX mixer.
All four negative-cut phase layers and positive RX(beta) mixers remain intact.
The original linear schedule spans zero to one radian, without pi scaling.
Native COBYLA counts trust-region iterations whereas source SciPy maxiter=40
counts function evaluations. Stop the callback before a forty-first execution,
select only actually evaluated parameters and report evaluation_budget and
nonconvergence. Do not fabricate an optimizer result or claim trajectory equality.
Exact local mean probabilities replace provider shot-noisy cost estimation;
final 1,000 seeded samples are actual draws outside the objective budget.
Named array readout replaces manual bitstring decoding. Independent full Circuit
states include absolute phase, bit order and clean workspace. Enumerated classical
optima are only comparison labels, never inserted into samples. Preserve native
local CPU execution; no remote, release, hardware or advantage claim.

Run from the repository root:

```bash
python -m examples.synthesis.tutorials.workshops.combinatorial_workshop.combinatorial_qmod_workshop_for_maxcut.overview.main
```

Open `walkthrough.ipynb` for all eighteen original teaching stages, actual
search and samples, network plots and independent Circuit checks.
See [WORKFLOW.md](WORKFLOW.md) for acceptance boundaries.
