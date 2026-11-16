# Bound-vector preparation workflow

The complete reviewed Notebook has two quantum programs. The first prepares the
probability vector `[0.05, 0.11, 0.13, 0.23, 0.27, 0.12, 0.03, 0.06]`, samples it
and reconstructs a probability array. The second constructs eight evenly spaced
real amplitudes from -1 to 1, normalizes them, prepares them, reconstructs their
complex state and removes its global phase for display.

`main.py` performs these steps with ordinary Python vector generation, native
`qfunc` definitions, `classical_inputs` and local FlagQuantum execution. Existing
synthesis-time binding snapshots vectors before capture. The public user passes
vectors to `prepare_state` or `prepare_amplitudes`; no gates or physical wires
are needed. No arbitrary host code runs inside a quantum function.

1000 seeded samples from executed named probabilities replace the cloud sampler.
A plain list replaces dataframe reconstruction. The amplitude program has one
named output register and a zero-workspace constraint, so its raw statevector is
the requested amplitude vector. Phase alignment is used only for the printed
presentation: the first raw amplitude is negative, and the displayed aligned
vector has its first amplitude positive, matching the source convention.
Verification compares raw complex states without removing global phase.

Independent dense-reflection Circuits verify two nonuniform probability vectors,
two signed amplitude vectors and the maximum 256-element vector. Acceptance also
checks literal/bound resource equivalence, immutable snapshots, bit order, zero
workspace, normalization, widths, storage ownership and resource rejection.
Parameterized Output helpers remain outside the existing admitted helper profile.
This is a complete local computational workflow, not cloud or hardware execution.
