# Native commuting-Hamiltonian HHL composition

Status: independent dependency probe passed; complete native acceptance pending.

The original HHL uses a two-bit solution, a four-bit signed all-fractional phase
counter, a signal bit, four commuting Pauli terms and a retained approximate
inversion network of 16 RY/CX pairs. Existing native QPE, Pauli product formulas,
within/apply and raw-word rotation tables express this program. No production
Compiler or public API change is required.

Native QPE repeats the first-order commuting evolution; the source instead
scales evolution time by each power. These operations are mathematically equal
because all four terms commute. Preserve the identity coefficient and absolute
phase under control. Preserve the source's approximate rotation coefficients;
do not substitute ideal reciprocal eigenvalues. Compile their exact parity
network into 16 raw-word Y angles using the existing select_rotation entry.
Counter numeric decoding remains signed word/16, independently of raw table
indexing. The source skip_control wrappers are inert at this uncontrolled root;
this example does not introduce a generic skip_control API or promise nesting
inside a new outer coherent control.

Acceptance compares every complex amplitude and all work with an independent
Circuit using analytic matrix exponentials, Fourier matrices and the original
16-pair inversion network. Verify the unchanged original plus two nonzero RHS
vectors; separately test the inversion table on coherent selector and nonzero
complex indicator states. Check signed decoding, bit order, invalid arrays,
checked inverse, and qubit/work/operation/depth/memory refusals. Report honest
resources: native repeated evolution need not match powered-source gate count.

Final fresh replay of all 216 originals and six approved revisions, immutable
evidence, catalog and ledger updates are required before credit. Native coverage
starts at 196/216; Notebook coverage remains 24/213. Classification is
`single_device_fast_path`. No integration PR or full remote CI before complete
216-program native acceptance.
