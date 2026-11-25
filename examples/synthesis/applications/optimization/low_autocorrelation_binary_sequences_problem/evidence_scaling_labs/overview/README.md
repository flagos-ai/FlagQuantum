# Low-autocorrelation binary sequences

Generate the original LABS energy for 13 spins, construct a three-layer QAOA
circuit, and compare three equivalent parity schedules in the CX/U3 basis.
Run the selected circuit through the source-sized 60-query numerical search
and report the sampled low-energy words.

```bash
python -m examples.synthesis.applications.optimization.low_autocorrelation_binary_sequences_problem.evidence_scaling_labs.overview.main
```

Open [the Notebook](walkthrough.ipynb) for the optimization trace, the observed
histogram, and the independent state and exact-cost reference.
