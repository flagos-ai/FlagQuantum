# Exact amplitude amplification workflow

This completes the local computational content of the source Notebook:
prepare good-state probability 0.07, perform exact amplification, sample and
assert success, and compare standard/tuned Grover trajectories in the good/bad
plane. It reuses the existing native three-round R/phase quantum composition;
no native higher-order amplification API is introduced.

```sh
python -m examples.synthesis.functions.amplitude_amplification.exact_amplitude_amplification.main --plot /tmp/grover-trajectories.png
python -m examples.synthesis.functions.amplitude_amplification.exact_amplitude_amplification.comparisons.circuit
```

The first command samples 1000 outcomes from actual executed named probabilities
and returns `{(1, 0): 1000}` for (target, auxiliary). The explicitly named auxiliary
and hidden workspace are zero. The Circuit check compares raw complex amplitudes
including absolute phase. The plot contains all four vectors of each trajectory:
the standard trajectory overshoots; the tuned trajectory ends at (1, 0).

Local seeded sampling replaces the cloud call; a saved Matplotlib figure replaces
interactive display. This is a local computational workflow, not a hardware or
cloud service claim. The retained source's reviewed code-cell hashes and these
adaptations are recorded in `provenance.json`.
