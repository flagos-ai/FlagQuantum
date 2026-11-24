# Select-swap QROM constructor and dependency disposition

Reference: complete `data_loading/select_swap_qrom.py` and direct QROMBase,
QROM, Partition, SwapWithZero, word-XOR and Adjoint dependencies at unmodified
Qualtran v0.7.0 commit bcb17bb25360ed68ce3bc084645db3e929799389.
The focused fixture verifies the three original Git blobs, Qualtran 0.7.0 and
actual Kickmix 0.1 Simulator. Original references are not patched.

## Ownership and interface

Selection, controls and scalar target words are THRU. Datasets and block cells
are row-major; bits are MSB-first. Targets may start nonzero. Forward dirty mode
borrows temporary banks and restores their entire coherent state; clean mode
owns zero-initialized banks. The actual original inverse allocates clean banks
from forward default Free, including when forward used dirty Allocate. Its
final Free retains the corresponding source dirty label. Physical pools track
the actual lease ownership. Source labels and actual pool flags are distinct.

The shared blocked_lookup_component owns existing typed selection partitions,
QROM calls with reserved scratch, routing and reassembly. It serves both clean
lookup and select-swap XOR lookup. Shared legacy_lookup_metadata reads their
common serialized fields. Neither helper introduces another runtime context,
IR type, registry or numerical implementation.

The select-swap component retains Load/Route/Copy/Unroute/Load and, for dirty
mode, another Route/Copy/Unroute to cancel borrowed bank contents. Adjoint
reverses component calls and the dataset order at copy/routing boundaries.
QROM is the same XOR involution with its forward measured cleanup, never a
reversed measurement stream. SwapWithZero retains its source self-adjoint
relative exchanges and reverses the exchange network. Existing word-XOR,
source-reserved unary scopes and CSwapApprox remain authoritative.

The explicit constructor and its Adjoint dependency retain recipes within one
library compilation. Unknown adjoints delegate only to an explicitly supplied
existing owner or refuse. Compilation from serialized concrete records works
with source imports blocked. Users retain the existing circuit.append(op)
execution path and native owned Program data.

## Comparable source evidence

Both concrete original examples are reconstructed from their complete metadata.
Serialized depths zero and one have identical reachable definitions. Original
Adjoint serialization can assign different IDs to its unused forward recipe;
the existing literal definition matcher ignores names and unreachable records,
while preserving every reachable port, operation, value and dependency. Every
supplied definition is still verified. Independent analytical models match the
original gate categories and QubitCount in both directions and bank policies.

Default and blocked versions of both original examples compare strict primitive inventories,
per-wire operands and feedback. The dirty inverse starts routing the last
dataset before its first QROM load, so allocation events occur in reversed
dataset order. Tests bind source bank arguments by semantic dataset identity,
not by assuming event order equals dataset order. No gate permutation or
workspace renaming is used to waive an operand mismatch.

Small one-dataset four/five-entry cases exhaust all raw measurement histories
with complex inputs, nonzero destinations, spectators and all borrowed-bank
basis patterns in coherent superposition. Banks and temporary work return to
their original states and no private leases remain live. Larger two-dataset and
multi-axis cases use coherent sparse execution over every valid address and
multiple destination/bank patterns. Whole-circuit erasure certificates verify
every measurement history for these default and blocked layouts. Existing raw source QROM channel
certificates and live-graph bank restoration evidence are reused. Quantum
primitive counts and peak leases are reported separately from analytical model
nodes in the resource receipt.

The previous clean lookup constructor/component tests run unchanged after
extracting the shared stages and metadata parser.

## Original defects and remaining scope

All four original symbolic examples fail unchanged serialization with
`Object arrays cannot be saved when allow_pickle=False`. Native shape-only
lookup declarations and model queries remain separate from this source codec.
Original missing high/low-selector and whole-bank finalization failures stay
explicit. Missing generic control of non-through And/And-adjoint remains a
source decomposition boundary. Shaped-target and QROM admission differences
remain in the central ledger. No generic constructor identifier or parent
acceptance is incremented for this concrete disposition.

## Acceptance receipt

The concentrated regression passes all 328 cases without skips. Local integration
passes 145 cases. The complete unchanged original test file, including its slow
classical and tensor cases, passes 27 cases with two original resource-count
xfails in the existing Cirq 1.6 environment. Strict typing covers eight files;
Black, Ruff, team scope, language, architecture, IR and protected API checks
pass. The dependency-policy output is identical to the 720 existing findings.
No protected public API or serialization vocabulary is changed.

For the blocked original multi-data example, clean and dirty modes emit 56 and
104 T-family primitives, 112 and 186 CX, two measurements and 22 peak physical
wires. The separate source models contain 48/96 T and two And nodes, with 22
logical peak wires. For the multi-axis example, clean and dirty modes emit
296/536 T-family primitives, 678/1036 CX, 14 measurements and 59 peak physical
wires. Their separate source models contain 240/480 T and 14 And nodes, with
59 logical peak wires. Counts agree in both directions; private live bits at
return are zero. These inventories remain separate from the model categories.
Detailed receipts are `g2-select-swap-constructor-{regression,local-integration,
original,quality,resources}-20261124` under the external reference ledger.
