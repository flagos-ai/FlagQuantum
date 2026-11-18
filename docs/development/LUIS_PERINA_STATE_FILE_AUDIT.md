# Complete Luis–Perina resource-state source

## Frozen source and ownership

Q-0624 maps the complete Qualtran v0.7.0 file
`phase_estimation/lp_resource_state.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389`, blob
`55466291aba7d239ad4992b3aa3e09b44e33afe2`. Q-0625 maps its complete tests,
blob `41b84e5cb8df63838ddbd843d2e4202bd1ad6c18`.

Algorithms owns `luis_perina_state.py`. LPRSInterimPrep maps to
LuisPerinaInterim and LPResourceState maps to LuisPerinaState. The shared window
base owns RIGHT-only fractional output descriptors. Interim `m` is big-endian
THRU data and `anc` is a THRU logical qubit. Preparation accepts clean outputs
through the existing named PREPARE binder. The caller uses
`circuit.append(state.bind().make_on(selection=phase, junk=(), workspace=work,
classical_bits=slots))`.

The caller owns n+4 clean work wires and n+1 reusable result slots. The first
work wire is the source ancilla, the second is the source flag, and the shared
reflection owns the remainder. All work returns to zero in every branch.
Interim uses no scratch, measurement or result slot. Core owns exact unitary
inversion. Compiler owns OnEach, CRz, predicate measurement and reflection.
No production code imports Qualtran or Cirq.

## Complete responsibility mapping

- Interim H on the word, H on anc, reversed word traversal with exactly
  `-2*pi*2**i/(2**n+1)` controlled Rz, bare anc Rz and final H are preserved.
- Default eps remains 1e-11 metadata. The source emitter does not use it to change
  these rotations. Generic adjoint uses the shared opcode inversion rules.
- The window's standard-deviation factory preserves ceil(log2(pi/eps)). Its
  signature reuses the complete shared fractional window descriptor.
- The full amplitude-amplification order is Ry, interim, CZ, interim inverse,
  inverse Ry, zero reflection with coefficient i, interim, Ry, X flag, X anc.
- The seven original direct call keys and weights map to the native source
  graph. Interim retains each angle-specific CRz call, including the original
  symbolic aggregation. All three original examples map to the same constructors
  and shared signature/graph drawing. Notebook responsibility is Q-0623.

## Execution and resource evidence

The complete original suite, including every slow case to n=13, reports
38 passed, 1 original strict-type skip, 2 original count xfails and 2 original
unregistered slow-mark warnings. The exact log is `g2-lp-state-source.log`.

Interim source capture matches every primitive, operand, angle and direction at
n=1,2,3,5,13. Full operators through n=5, prepared tensors through n=13 and
complex coherent forward/inverse restoration are checked. The n=13 test avoids
forming the dense operator. Replacing the interim emitter with the captured
source stream preserves the complete measured preparation consumer.

At n=1,2,3, every forward measurement history and every independent joint
forward/inverse history is enumerated. Unnormalized complete complex amplitudes
match the original resource-state tensor tensored with a complex spectator and
clean work. Exact inverse restores that input, including its global phase.
Native execution is bounded to these widths for exhaustive dynamic histories.
The complete source tensor slow suite covers through n=13.

The actual positive-width interim inventory is n+2 H, 2n phase, 2n CX and one Rz.
The actual full preparation inventory is three Ry, 6n+9 H, 6n+1 phase, 12n+6 CX,
three Rz, n+3 CZ, 2n+4 identity, 4n+12 X, 2n+2 T, 2n+2 T-dagger, n+1 S,
n+1 measurement and n+1 reset. These identities are execution evidence and are
asserted at all exhaustive widths.

Independently, the declared QEC interim model is 3n+2 Clifford and 2n+1 rotation.
The concrete full model is n+1 And, 14n+21 Clifford, 6n+6 rotation and n+1
measurement. The original symbolic model is instead n+1 And, 10n+13 Clifford,
6n+6 rotation and n+1 measurement. The shared symbolic predicate model retains
that distinction. Every category is compared to the untouched source for all
original resource widths 1,3,5,7,9,11,13 and symbolic n. Original legacy T
formula tests also pass in the complete source suite. Physical inventories are
not presented as declared model counts. Inventory and extension preflight are
in `g2-lp-state-inventory.log`.

## Original defects and uncovered boundaries

The original strict check skips the same QFxp/QAny boundary between allocation
and interim preparation. Its original two count autotests remain xfailed.
The original symbolic QubitCount query raises for both classes. Native code
exposes the symbolic gate graph and concrete physical storage separately.

The source's nonreal zero-reflection adjoint returns the forward reflection.
Consequently the complete original LP adjoint tensor is the negative of the
mathematical bra at n=1 and n=2. The native explicit reverse emitter uses the
shared mathematical inverse and is tested separately. This source defect
remains an open parity boundary.

The generic SDK snapshot of the full LP source chooses a different MultiControlZ
synthesis with opaque matrix leaves and fewer allocated wires. It differs from
the source-specific measured predicate call chain used by the existing native
reflection. Interim primitives and the source call/model boundaries are exact.
Full SDK-default primitive inventory equality is not certified. Prior reflection
all-branch and source-declared-graph evidence is reused, not relabeled as a new
SDK-default synthesis comparison. Q-0625 remains open for these specific source
and execution-path boundaries. Zero and unresolved widths retain declarations
but have no native physical admission in this batch.

## Batch checks

The concentrated window, PREPARE, reflection, controlled-rotation, LP and GQSP
regression reports 310 passed. Black, Ruff, strict mypy for both new Python
files, Algorithms scope, architecture, multi-level IR and protected API checks
pass. Logs use `g2-lp-state-` in the external reference workspace. Kickmix 0.1
and its native Simulator were verified before execution. Q-0624 closes the
complete source responsibility mapping. Q-0625 retains the boundaries above.
