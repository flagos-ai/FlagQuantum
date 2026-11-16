# Preparation error-bound sweep

The complete source Notebook generates a cubic trend on 256 evenly spaced points,
adds a seed-one Gaussian perturbation, mean-centers and normalizes it, then
synthesizes ten requested amplitude L2 error bounds from zero through 0.3.
It reports depths and CX counts and plots both against the requested bound.

`bound_sweep.py` implements that complete local workflow using native qfunc,
classical real-array and bound parameters, execution and resource reports.
The retained Qmod program has a different 64-element vector; `build_program()`
in `main.py` preserves that vector and its separate source identity. Executing
`main.py` runs both the retained program and the eight-bit Notebook sweep.

```sh
MPLBACKEND=Agg python -m examples.synthesis.tutorials.technology_demonstrations.approximated_state_preparation.main
```

The plot is saved as `native_amplitude_bound_sweep.png` in the current directory.
`bound_sweep.run_example(output=...)` accepts another path. The native quantum
function contains no gate design, physical wires or bitstring decoding.

The current compiler prepares these real amplitudes exactly. All ten bounds
therefore use 8 qubits, 509 operations (255 RY and 254 CX), depth 502, and zero
workspace. The measured amplitude L2 error is 1.57e-15 and probability L2 error
is 2.96e-16. There are no approximation savings. The plot states
this explicitly. Depth and CX counts are native instruction-basis measurements;
they are not advertised as equivalent to Classiq's custom CX/U transpilation
metrics or cloud optimization timeout. No hardware/backend result is fabricated.

An independent eight-bit Householder Circuit checks each complete raw complex
statevector, including signs, absolute phase and ordering. Acceptance checks
amplitude and probability L2 error, every requested bound, zero workspace, resource
rows, the saved plot and preservation of the six-bit source program. Reviewed
Notebook code-cell hashes and the source pin are recorded in provenance.
