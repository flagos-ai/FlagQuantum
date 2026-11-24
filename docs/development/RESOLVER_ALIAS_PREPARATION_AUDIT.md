# Dense and sparse alias constructor dependency disposition

Reference: complete `state_preparation/state_preparation_alias_sampling.py` and
its unchanged tests at Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`. Both source Git blobs match the
locked manifest. Fixtures require Qualtran 0.7.0 and actual Kickmix 0.1 Simulator.

## Ownership and exact source data

Compiler preparation now owns the shared five-call typed graph: uniform
preparation, sigma Hadamards, QROM, comparison and controlled swap. The existing
Algorithms alias_preparation_program keeps its signature and table/shape
adaptation, then delegates using Core Signature and QromTable data. Compiler
does not import an Algorithms oracle or duplicate its numerical implementation.
Existing scalar, symbolic and application consumers use that same graph.

Compiler formats reads the original selection registers, alternate/keep data,
probability width and normalization; sparse metadata additionally contains its
index table. Selection width and iteration bound are explicit original fields,
never inferred from table maxima. Normalization does not change this quantum
graph. Existing independent model owners retain their own normalization input.
Typed source ports are preserved without replacing sparse QUInt/QAny with a
bounded descriptor or renaming a dense custom selection port.

Selection and logical junk are MSB-first THRU outputs. Logical sigma, alternate,
keep, flag and sparse row words remain caller-owned and may be nonzero or
entangled. Only each child's private work is erased. Exact inverse reconstructs
the owned component calls in reverse order; measured primitive streams are not
reversed. Scoped constructor recipes also supply the exact inverse and selection
metadata to the existing reflection constructor, closing its automatic concrete
alias dependency. Unknown Adjoint children require an explicit owner or refuse.

## Acceptance scope

All five original examples are inspected. Three concrete examples resolve in
both directions and at serialization depths zero and one. Complete recursive
calls and operands match the original; bound primitive circuits equal the
existing certified direct owner. Source-free compilation covers the original
alias-PREPARE reflection without a bespoke test factory or injected inverse.

Small dense/sparse constructors compare complete complex operators on arbitrary
logical inputs, including nonzero logical junk. The existing all-history erasure
certificate and full basis-column checks verify private work cleanup. Custom
dense names and sparse bounded, unsigned and opaque storage are covered. Existing
source primitive/feedback, complete inverse, append, copy and JSON evidence runs
through the unchanged Algorithms consumer in regression.

## Original boundaries and separate estimates

Both original symbolic examples fail unmodified Shaped serialization in both
directions. Existing native shape-only graphs remain available separately; a
failed original codec is not labeled a successful constructor. Dense unbounded
selection gives original n_coeff zero and fails decomposition. Sparse unbounded
selection executes using support length and physical width, but its declared
zero domain is retained. The source sparse body hardcodes `selection`; renamed
selection is refused rather than silently repaired.

Singleton, shaped/one-sided selection, other dtype and malformed QROM admission
boundaries remain explicitly separate. The original alias reflection simulation
xfails, nonreal reflection self-adjoint defect and uncontrolled scalar Proposal
119 boundary are unchanged. No generic constructor or parent-file acceptance is
incremented for unverified metadata admission. No protected API or schema changes.

Primitive T/Tdag, rotations, measurements and physical leases remain separate
from source CSwap/And/Clifford/rotation and logical peak estimates. Private lease
release alone is not proof of numerical cleanup of arbitrary reflection junk.
## Acceptance receipt

Final concentrated regression passes 330 cases without skips. Affected local
integration passes 145 cases. Strict typing covers four files; Black, Ruff,
Compiler and Algorithms scope, language, architecture, IR and API checks pass.
The dependency-policy output remains exactly the existing 720 lines. The
complete unchanged source file has 16 passing tests, three original skips and
three original xfails; its separately executed original notebook also passes.

The original dense example emits 42 T-family primitives, 117 CX, seven
measurements and 19 peak physical wires in both directions. Its source model
retains two CSwap, seven And, 80 Clifford, seven measurements and 19 logical
peak wires. Both original sparse examples emit 56 T-family, 144 CX, seven
measurements and 25 peak wires in either direction; their model retains four
CSwap, seven And, 91 Clifford, seven measurements and 25 logical peak wires.
The source total_t_count estimates are 36 and 44 respectively because that
query charges four T per CSwap; the exact emitted swap decomposition uses seven.
The source primitive/operand checks retain the exact decomposition. The model
is not relabeled as a measured primitive count.

Automatic original alias reflection emits the previously accepted 242 T-family,
630 CX, four Rz, 50 measurements and 41 peak wires. Its separate source model
retains six CSwap, 50 And, 461 Clifford, four rotations and 41 logical peak
wires. Private leases are released. The original reflection simulation xfail
remains separate from this structural/resource evidence.

Receipts use `g2-alias-constructor-*-20261124` under the external reference
ledger. They include exact original blobs, complete source/notebook outcomes,
source-free reflection dependency, full complex operators with all-history
private cleanup and separate seven-row resource inventories.
