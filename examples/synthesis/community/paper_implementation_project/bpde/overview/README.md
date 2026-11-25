# Bayesian phase difference estimation for hydrogen

Generate the full 30-distance STO-3G hydrogen curve and fresh displaced
Hamiltonians. Execute the original short and long readout scans and update
the five-point Bayesian phase-difference estimate.

The local example environment needs NumPy, SciPy, and Matplotlib in addition
to FlagQuantum.

```bash
python -m pip install scipy
python -m examples.synthesis.community.paper_implementation_project.bpde.overview.main
```

Open [the Notebook](walkthrough.ipynb) for all model parameters, all six
figures, the sampled update history, and the final energy derivative.
