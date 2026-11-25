# iQuHack Gaussian reference solution

Open [walkthrough.ipynb](walkthrough.ipynb) or run the original-scale
Gaussian preparations locally.

```bash
python -m examples.synthesis.community.hackathons.iquhack_2025.gaussian_reference.overview.main
```

The first reference prepares all 64 signed coordinates directly. The
next two circuits retain eight and four data qubits, respectively. They
load the Gaussian amplitudes on an indicator qubit and apply the
source's exact-amplification schedule with attenuation and reflection work.

The archived overlap formula evaluates an evenly spaced grid ending at
`1`. The signed lookup extends to `2 - 2**(2 - width)`. The walkthrough
runs the archived formula with all work wires retained and measures the
resulting cleanup error. It then recomputes the overlap from the actual
lookup grid and checks the corrected circuit at both original widths.
A two-dimensional Grover calculation independently predicts the leaked
probability and the complete clean data amplitudes. The figure and single
seeded draw use the validated eight-bit preparation.

The team export cell remains disabled as in the original. The export
helper requires an explicit team name and rejects any circuit with
measured dirty work qubits.
