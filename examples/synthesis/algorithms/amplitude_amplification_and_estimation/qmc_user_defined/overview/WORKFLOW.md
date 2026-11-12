# Complete workflow acceptance

Review all13 executable source cells and the surrounding mathematics. Keep
three stages: the probability/linear-Y loader, the complete Grover operator
with a zero-state reflection and constant global phase, and three signed
fractional QPE bits. Preserve full-precision p(x)=x/28, slope0.5, offset0.4,
the seven-qubit limit, native resource reports, finite sampling, signed phase
histogram and measured-vs-exact amplitude assertion below0.1.

Compare each complete physical state against an independent Circuit assembled
from conditional rotation matrices, a full prefix-probability preparation,
zero-state and indicator reflections, controlled matrix powers and a dense
inverse Fourier transform. Compare absolute phases without alignment. Also
check two nonzero alternative angle/distribution profiles. Verify every column
of the full loader on all16 computational basis inputs, because its inverse
inside Grover depends on the whole unitary, not only its prepared state.
Check logical bit order, exact indicator joint probabilities and zero work.
Dropped data in QPE is retained rather than assumed zero.

Compute source probabilities afresh in full precision. Keep the frozen Qmod
counterpart's rounded constants unchanged and its fullstate contract separate.
Sample2048 outcomes with predetermined seed123 from the actual native phase
marginal. Retain all eight signed bins including zeros. Select the largest
observed count without the exact answer, then compare to an independently
computed classical expectation. Invalid probability length, negativity,
normalization, nonfinite angles and resource budget violations refuse.

Execute all four actual Notebook code cells and both CLI launchers. Inspect
the histogram, its signed fractional axis and shot-count label. Saved source
Notebook outputs remain empty. Run216original and six-approved revision
replays. No hardware,analytic confidence,optimizer equivalence,speedup or
release certification claim is made.
