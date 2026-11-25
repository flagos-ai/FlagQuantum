# UP-06 checked exact amplitude amplification

Private Qmod adaptation admits a bounded exact_amplitude_amplification profile:
one whole initialized-zero unsigned target of one through seven bits, a closed
single-target sign oracle, and a closed product-RY preparation. The first scalar
is the positive amplitude of the complete marked subspace, not its probability.
It must agree with independently enumerated preparation mass to absolute 1e-12.
Zero, nonfinite, out-of-domain, inconsistent and over-budget parameters refuse
before owned effects. Sources and parameters are never silently corrected.

Choose k=ceil(pi/(4*asin(a))-1/2), capped at 32, and theta=pi/(4*k+2).
A separate ancilla starts with zero amplitude sin(theta)/a, obtained by RY.
Amplify the intersection of the source-marked subspace and ancilla zero using
k ordinary positive Grover reflections. The exact output has positive good-state
amplitudes and ancilla zero. Retain that ancilla as hidden physical storage and
count it in all budgets; do not assert an unproved release or reuse. Zero-round
amplitude-one preparation needs no ancilla. Every input bit and the ancilla must
remain within the existing eight-input local predicate bound. Reuse existing
RY, indexed phase, predicate normalization, zero reflection and Compiler kernels;
no new owned opcode or physical template is added.

Product probability vectors use per-bit marginals and an exhaustive product
factorization proof. Existing exponential preparations reuse the stable RY
angles. Nonproduct and complex preparations, quantum captures, consuming or
Output callbacks, aliases, mutation before invocation, recursive callbacks and
unsupported scopes refuse. Generic inplace_prepare_state in this slice shares
that checked product profile. Existing real amplitude initialization keeps its
fresh-output semantics. Only root EAA calls are admitted in this first slice.

Accept unchanged exact_amplitude_amplification.qmod with independent public
Circuit and analytic full vectors, including absolute phase, bit order and
clean hidden ancilla. The unchanged partial_exponential_state_preparation.qmod
supplies 0.996625424765961 while its five-bit exp(+x/2) distribution and x>=21
predicate have amplitude approximately 0.9979545784437375. Official Classiq
API documentation warns that inaccurate amplitude prevents exact output and
correct auxiliary cleanup. Preserve this original as a semantic refusal, not a
capability success. A separately labelled corrected diagnostic copy can test
the same path, but does not add original coverage. Test amplitude/round bounds,
nonzero input refusal, phase-oracle validity, callback boundaries and actual
resources. Recheck all 216 originals and preserve earlier AO/PF/resource evidence.
No integration PR/full CI before complete corpus acceptance.

References: https://docs.classiq.io/sdk-reference/qmod/functions/open_library/amplitude_amplification
and https://docs.classiq.io/sdk-reference/qmod/functions/open_library/state_preparation.
The algorithm follows exact Grover rotation with an amplitude-reducing ancilla;
implementation is independently derived rather than copied from a foreign SDK.

The corrected diagnostic copy additionally needs `x += -18`. Admit an unsigned
literal constant offset with absolute value at most 256, reduce it modulo the
known register width to the shortest signed representative, and compose the
existing +/-1 reversible increment kernel. No separate adder template is added.
A zero offset still emits an identity read so initialization is checked. All
emitted effects and physical work count against existing budgets. Dynamic or
quantum offsets and signed/fractional targets remain outside this extension.
Test every small basis word, wraparound, coherent phase and cleanup. This shared
constant-offset extension does not repair the original's amplitude mismatch.

Owned accounting extension: qreg.hidden may carry the optional attribute
workspace=True, which the register verifier checks as exactly the boolean True.
Absent means existing hidden semantic data and preserves prior accounting.
Marked storage remains physically allocated after qreg.drop; drop only removes
its logical view. Register scheduling includes marked widths in peak_workspace,
plus simultaneously needed existing Compiler scratch. This closes the observed
max_workspace=0 omission for an algorithm's retained internal ancilla. No new
opcode, physical realization or unproved early release is introduced. Test the
owned marker validation, retained allocation, exact candidate workspace count
and synthesis budget refusal independently from source-level acceptance.

A close scalar alone is insufficient near amplitude one or a rotation boundary.
Admission also proves the residual bad-subspace amplitude after the specified
rotation is at most 1e-12, using the enumerated actual mass and the supplied
scalar. Otherwise refuse rather than claiming exact cleanup. In particular,
a supplied amplitude one must not hide a small but nonzero bad population.
