# Complete unary iteration session source mapping

## Interface and ownership

This batch closes K-0237, K-0238 and K-0239, the complete locked Kickmix
`py/circuit/subs/unary_iteration.pybind.cc`, its header and original test file at
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`. The implementation, header and tests
contain 514, 76 and 807 lines respectively. All 45 direct dependency closure
blobs match the immutable inventory. The unchanged upstream suite passes all
59 tests in the locked extension.

Compiler owns context/iterator lifecycle because it emits physical operations
and manages compile-time workspace. The private `UnaryIterationPass` consumes
an existing operation list, `RegisterWorkspace`, a native big-endian address
word, an optional step-one range and an existing quantum/classical/Boolean
control. It delegates every physical move to `UnaryAddressCursor`. External
source addresses are little-endian and are converted once at the test or
adapter boundary. The HMR result slot is explicit and distinct from an external
classical guard. Existing physical program lowering and `circuit.append(op)`
remain the construction path. Runtime and numerical Simulation are consumers.

The caller owns address, control and body-output wires. The pool owns allocation
lifetimes. An open pass owns one clean temporary lease. False control reserves
one clean zero wire, visits nothing and emits no operations itself. Other passes
reserve `max(1, len(address))` wires. Classical control scopes the complete pass
with existing push/pop records. Explicit close emits cleanup and releases the
lease. Destruction emits no operations and leaves allocation outstanding, as in
the source. Both shallow and deep copying are refused to preserve unique
ownership.

## Complete definition map

| Source responsibility | Native owner and preserved behavior |
| --- | --- |
| `parse_address_values`, `make_unary_iteration` | Constructor reuses operand binding, validates address and width, admits None or step-one ranges and retains arbitrary Python integer endpoints. |
| Header state and `address_space_size`, `is_open` | Address tuple, range, control, lease, accepted cursor, external position and independent iterator-next state. The sentinel is `1 << width`. |
| `check_open`, `open_session` | Explicit lease acquisition, repeated-pass reuse, capacity failure, quantum/classical/false control dispatch and rollback of invalid cursor construction. |
| `close_session`, destructor | Idempotent close, physical cleanup, balanced classical scope and lease release. Destruction has no gate or release side effects. |
| `match_qubit`, `cur_address_value`, `move_to` | Match belongs to the first leased wire. External huge or negative positions remain observable while physical out-of-range positions park at the sentinel. Index-protocol inputs are accepted. |
| `next`, `__iter__` | Iterator-next state is independent of explicit seeks. Iteration may restart before its first yield, rejects restart during a yielded iteration, and leaves the context open on exhaustion. False control exhausts immediately. |
| `__enter__`, `__exit__`, `close` | Reusable contexts, exception cleanup and explicit close share the same lifetime owner. |
| `repr`, registration methods and documentation | Private native class representation and existing component construction replace foreign namespace registration. All registered operations are mapped above. |
| Original tests | Unchanged source suite plus source-generated complete native scenarios and replacement of accepted amplitude and normal-append consumers. |

## Executed evidence

`tests/unit/fixtures/generate_unary_iteration_session.py` verifies extension
version 0.1, `Simulator` availability and SHA256
`5e849075b49d895cca761fae864c8561b54362a6158fbfcabe7e9eb6e96d789e` before
collecting observations. Its 112 scenarios cover widths 0 through 3, true,
false, quantum and classical controls, full sweeps, out-of-range ranges, huge
seeks, early breaks, exceptions, repeated passes and empty ranges. A nested
traversal shares four clean wires between outer and inner passes. Regeneration
of `tests/unit/data/unary_iteration_session.json` is byte-exact.

Every scenario compares the complete physical stream, including source block
markers, control scopes, reset, HMR, feedback operands and body calls. Only
register declaration metadata is removed at the explicit port-binding boundary.
Visited values, reported positions, source allocation counts and complete lease
reacquisition are checked separately. No existing reference code is modified.

Twelve accepted full-complex, all-reset/all-HMR-history consumers are replaced
with session construction without changing their assertions. Widths 0 through
2 cover controlled and unconditional seeks, repeated and invalid addresses,
exact branch amplitudes and zero workspace. Classical guard enabled and disabled
paths separately enumerate every small raw measurement history. The normal
`examples/algorithms/unary_address_traversal.py` consumer also runs with the
session emitter substituted behind its existing construction function.

The 59 original tests run against upstream. They are distinct from the native
scenario tests, and are not claimed as an unchanged native replacement suite.
The concentrated batch passes 1387 tests and 121 local integration tests. Strict
typing passes for all three Python files. The production session has 100 percent
statement and branch coverage. Black, Ruff, Compiler scope, architecture, IR and
public API checks pass. Dependency findings remain at the identical 720-entry
baseline. Receipts are in `ftqc-reference-env/g2-unary-session-*20261123.*` beside
the worktree.

## Resources and adaptations

Actual source primitives are compared by complete stream equality. The session
adds zero quantum primitives, measurements or work beyond the accepted cursor
and the explicit body. A two-address-bit positively controlled sweep has actual
X 6, R 3, CCX 3, CX 7, HMR 3 and CZ 3, totaling 25 physical records. Existing
lowering expands that stream to X 9, H 6, reset 3, CCX 3, CX 7, measure 3, CZ 3
and one classical initialization. These are separate physical and expanded-IR
accounts. Analytical model estimation is absent for this orchestration layer
and is recorded as None.

Source pointer identity, native extension registration and source-branded repr
are replaced by existing native operand, lease and class identities. Native
validation rejects duplicate or overlapping wires before physical emission and
requires distinct guard/result slots. Capacity and lifecycle exceptions use
native vocabulary. Invalid raw pointer states are outside the executable
comparison. A callback must preserve the live address, control and match wire.

The original cursor generator fingerprints and all-branch evidence are reused
and rerun. Existing SIMD runtime/performance gaps, other unclosed source files,
upstream defects and pending Proposals 116 through 119 remain separately listed
in the main ledger.
