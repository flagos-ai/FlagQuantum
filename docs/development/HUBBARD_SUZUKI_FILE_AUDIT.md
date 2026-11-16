# Hubbard second-order Suzuki file audit

Q-0443 and Q-0444 cover the two Hubbard Suzuki factories and their original
tests from Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`.
The two frozen Python files matched their Git blobs. The original
non-notebook cost test passed.

The ordinary and HWP factories share one shaped through register with
two opaque side-width words, as declared by the interaction template.
The native source cost uses that signature as its parent width.
The three frozen templates retain their zero angles, coupling parameters,
positive or negative color assignment and synthesis precision.

| Factory option | Application order and coefficients |
| --- | --- |
| Complete step | Interaction 1/2, pink 1/2, gold 1, pink 1/2, interaction 1/2 |
| Stripped step | Pink 1/2, gold 1, pink 1/2, interaction 1 |

The source expands each layer at angle `2 * coefficient * timestep` and
passes both spin words through each layer in the same order. Twelve
reference cases verify the signature, ordered operands, template and
expanded arguments, exact call counts and resource categories, and peaks.
The package's ordinary source count assertion for side eight is
`12 * length**2` direct T gates and `5 * length**2` rotations.

The native `HubbardSuzukiStep.make_on` binds row-major spin arrays of
`length**2` sites, with the physical rotation convention supplied by
the on-site and hopping emitters. Both schedules and their adjoints
match the exact instruction stream obtained by chaining the respective
standalone layers on deliberately permuted operands, with their feedback
slot mapped to the shared Suzuki slot. The independent two-by-two
Jordan–Wigner reference checks the complex matrices for all four ordinary
schedule and adjoint combinations. For HWP the tests cover all four
histories per layer, checking the full phase and restored work before the
subsequent layer. They also check an end-to-end forced history of each
complete or stripped step and feedback slots after an existing measurement.

| Fixture | Resource evidence |
| --- | --- |
| Native 2×2 complete ordinary | 48 emitted T-family instructions, eight wires |
| Native 2×2 complete HWP | 88 emitted T-family instructions, ten measurements, three work wires |
| Frozen 4×4 complete HWP model | 912 T equivalents, 70 measurements, peak 19 logical qubits |

Q-0443 closes at the factory boundary. Its child operators expose
declarations without frozen gate-level bodies. Their phase, width,
and synthesis differences remain recorded in Q-0437 and Q-0440.
Q-0444 retains the outstanding notebook check. In the pinned QPE notebook,
SciPy passes a one-element array to the objective at cell 22, and the
scalar rotation-cost calculation rejects that value. The frozen
reference remains unchanged. The native notebook work is tracked under
Q-0442.
