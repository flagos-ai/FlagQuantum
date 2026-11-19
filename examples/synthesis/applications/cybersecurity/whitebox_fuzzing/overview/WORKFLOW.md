# Source mapping and acceptance

Source code cell 4 defines the prioritized classical branches. Native `foo`
preserves those branches exactly. Cell 6 draws 500 x and 500 y values using
RandomState(3) and plots their output frequencies; the original half-open 0..63
range is preserved. Cells 10 and 12 construct the predicate phase oracle on two
six-bit fields and request width 25. Native code reuses the existing parent
`mark` helper and names those fields `x` and `y` without changing their values.

Cell 18 reports the source iteration recommendation 36, based on two marked
items. The truth predicate actually marks six pairs. Cell 21 runs ten Grover
iterations, and cells 23/25 synthesize and show that program. Native code retains
the original ten rounds and width 25, reporting actual resources locally in
place of a hosted viewer. No claim equates the source heuristic to the optimal
iteration count. Cells 27/29 draw 500 observations and annotate each row with
`foo`; cell 30 plots their count-weighted branch frequencies. Native sampling
uses seed 1913 where the source supplied no quantum seed, and named probabilities
avoid user-visible physical indices or bitstring decoding.

Acceptance exhausts the predicate on 4096 inputs and checks branch priority,
threshold equality and modulo residues. Verify classical random draws and
500 actual counts, seeded quantum repetition and the marked frequency against
independent Grover probability. Independent Circuit uses separately constructed
truth-table phase signs and diffusion, with its own clean AND ladder. Compare
complete absolute states at 0/1/2/10 rounds, bare-oracle signs and all hidden work
amplitudes. Preserve the source 6+6 bit problem and do not shrink it for tests.

Both rendered bar charts must be inspected for readable English labels and
correct count weights. Credit requires fresh-kernel execution of all 11 native
cells, strict quality checks, frozen 216+6 replay and the required full local
default-tier comparison. The existing parent quantum example and source
fixtures remain unchanged. No provider credentials, Qmod loading or remote jobs
are part of this native workflow.
