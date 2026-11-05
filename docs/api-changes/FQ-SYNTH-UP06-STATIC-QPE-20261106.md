# UP-06 bounded static QPE and controlled unitary composition

## Ownership and scope

This contract extends the existing private construction representation. Compiler
owns matrix validation, registered-gate synthesis and controlled composition;
Ecosystem owns source callbacks, fixed-point metadata and literal binding.
No stable export, public signature, numerical kernel or external SDK is added.
The path classification is single_device_fast_path. The immediate source targets
are quantum_counting_qpe.qmod and the two qpe_for_matrix/unitary_matrix originals.
Other QPE callers retain independent blockers and receive no coverage credit.

## Compiler contract

Preserve the existing one- and two-qubit unitary decomposition and cost baseline.
Admit bounded three-qubit matrices through genuine registered-gate decomposition.
Use deterministic complex Givens elimination and Gray-path two-level synthesis,
including every diagonal and global phase. Check reconstruction against the
original matrix at absolute tolerance 1e-12 before accepting the decomposition.
Do not execute a circuit or import Runtime, Simulation or Algorithms to certify it.
No arbitrary dense instruction is emitted by this compiler path.

The existing qreg.unitary may additionally carry power, a plain integer 1..128,
and control_index, one numeric bit index into a final initialized control operand.
The matrix and optional data indices describe only the data operands. Validate
the original matrix before taking its power. Require initialized writable data,
an initialized control, and no control/data storage overlap; Const controls are
permitted. Controls apply to the complete decomposition, including phase repair.
No matrix normalization, polar correction or approximate source repair is allowed.
Classical matrices remain at most 8 by 8; generated gates obey ordinary planning
budgets. Invalid attributes, malformed matrices and unsupported widths fail closed.

## Source contract

Ordinary qpe accepts a zero-argument static lambda over initialized surrounding
data and a fresh allocated all-fractional phase counter of at most eight bits.
Expand H preparation, controlled powers 2**i for numeric counter bit i, and the
existing inverse QFT. Reject counter capture, allocation, release, dropped data,
Output escape and unsupported callback effects. Control global phase explicitly:
a callback's global minus sign is observable on the phase counter.

Resolve literal global integer width constants without spelling-specific rules.
Accept finite bounded real[][] matrix constants and existing complex literals;
verify unitarity at use. Infer all-fractional phase metadata from typed allocate.
Signed phase readout uses the existing two's-complement convention and preserves
the same physical circuit. Partial fractional arithmetic is outside this slice.
No unsupported definition or zero-iteration case may bypass validation.

## Acceptance and migration

Compiler tests cover every basis input, coherent complex inputs, global phase,
asymmetric wire order, controlled superpositions, powers and malformed bounds.
References use independently specified Circuit matrices, never the production
synthesis. Existing one/two-qubit and corpus resource rows remain unchanged.

For each newly accepted original retain the source hash and compare the complete
complex statevector to an independent controlled-matrix-power and inverse-Fourier
Circuit, including absolute phase and fixed-point decoding. Check workspace,
resource rejection and alias/escape rejection. Re-audit all 216 unchanged originals;
count coverage only after independent original execution acceptance. Keep both
known invalid originals separate, without normalizing or modifying their sources.

A runnable scalar native qfunc comparison must use existing public operations;
do not advertise native array QPE or library exports that have not been implemented.
Cross-domain delivery lands this contract first, followed by domain-owned commits.
