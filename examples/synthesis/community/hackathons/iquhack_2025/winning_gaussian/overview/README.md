# Quantum Tree Gaussian preparation

This is the native FlagQuantum walkthrough of the winning iQuHack
two-stage solution. Open [walkthrough.ipynb](walkthrough.ipynb), or run

```bash
python -m examples.synthesis.community.hackathons.iquhack_2025.winning_gaussian.overview.main
```

Stage one initializes eight qubits with the source's factorized rotations,
small controlled rotation and truncated Fourier schedule in native qfunc.
The visible logical program uses a borrowed one-bit helper, static loops and
pairwise real phases. Synthesis handles physical ports and controlled lowering;
the original thresholds and fitted angles remain explicit classical inputs.
It compares all
256 amplitudes against an independent bit-index reference and compares the
signed probability grid with the normalized Gaussian target. The output
includes one actual seeded sample and the mean squared error.

Stage two constructs the same schedule at the original 127-qubit resolution
and reports the emitted CX/U3 circuit resources. It is a compilation-only
stage. Both stages save round-trip checked native circuit IR JSON files.
This large stage remains Circuit-based and is not claimed as qfunc execution.
The resource counts describe the local basis conversion and the common
phase introduced by that conversion is recorded on the smaller stage.
