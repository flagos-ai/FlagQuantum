# Native Pauli evolution over checked static views

The standing authorization for the complete native example library covers this
bounded extension of existing suzuki_trotter/qdrift targets. It changes no public
signature, export, IR operation or Runtime route. Classification is
single_device_fast_path; implementation is authorized, release is not.

A target may be a nonempty static QArray slice, a mutable-helper array view or an
explicit list/tuple of scalar bit locations. Every location must be distinct
and belong to one initialized writable register. Observable index zero denotes
the first target element, regardless of its physical position or numeric
significance in the owner. Capture maps each local Pauli index to its owner's
logical bit index and reuses checked qreg.pauli_rotation. It preserves axis and
term order, absolute identity phase and deterministic qDRIFT sampling. Empty,
repeated, out-of-bounds or cross-owner lists refuse before lowering; immutable
borrowing and uninitialized owners still refuse through shared effect checks.

Existing finite angle, operator, term, order, repetition and resource limits
remain. Small-owner compact products operate on the embedded owner space;
untargeted bits retain their identity action. Larger owners retain streamed
Pauli rotations. Zero-time calls retain an initialized-storage read. No new
controlled or general callable transformation is admitted.

For example, a root with allocate(4,q) can call
suzuki_trotter(OPERATOR, time, 1, 1, [q[3], q[1]]), where
OPERATOR=0.5*(fq.X(0)@fq.X(1)). The first Observable factor acts on q[3] and
the second on q[1]; q[0] and q[2] remain spectators. The same effect can be
captured inside a plain QArray helper borrowed over q[1:3].

Acceptance preserves the original Fermi-Hubbard eight-bit program, including
all ten Givens matrices and their ordered pairs, four zero-time exchanges,
four contact phases, positive/negative exchange times and num_iter=1/2.
Native, immutable original and an independent Circuit must agree in complete
complex amplitudes, including global phase and bit order. Asymmetric Pauli
supports over reversed/nonadjacent views, coherent spectators, identity phase,
both product orders, fixed-seed sampled sequences, helper borrowing, aliases,
initialization/Const errors and actual resource budgets receive separate tests.
All 216 original source hashes and frozen rows are replayed for each accepted
batch. Native quantum and complete Notebook coverage stay separate.
