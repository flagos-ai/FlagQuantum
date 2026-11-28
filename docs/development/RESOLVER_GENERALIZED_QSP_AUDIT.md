# Generalized QSP constructor and exact child ownership

Reference: complete unchanged Qualtran v0.7.0 `qsp/generalized_qsp.py`,
commit `bcb17bb25360ed68ce3bc084645db3e929799389`.

## Interface recorded before implementation

Simulation owns complementary-polynomial and phase-factor numerics. Ecosystem
injects its existing qsp_phase_factors into Compiler constructor adaptation;
no numerical phase recursion or external object enters Compiler. The factory
consumes the surviving U/P/Q/negative_power/precision fields, preserving the
original ascending coefficient order and precision/(len(P)+1) allocation.

The shared phase component takes explicitly owned native ordinary U and exact inverse
Programs, the actual source-selected positive controlled inverse, the existing
negative forward predicate Program, and ordered SU2 Programs. All U target
ports are caller-owned THRU registers, with their original names, dtypes and
shapes; `signal` is a leading scalar QBit and cannot collide with a U port.
Packed bits stay MSB-first. Explicit pair owners supply helpers, private work
and measurement erasure. A signature alone never supplies control or inversion.

The root calls the first SU2, then a controlled inverse while negative power
remains or a negative-controlled forward U otherwise, interleaving later SU2
calls. Surplus inverse powers call bare exact U inverse afterward. It reuses
the existing SU2 scalar/Rz/Rx/Rz constructor and privately owned scalar carrier;
no new scalar opcode or pending Proposal119 behavior is introduced. Negative
forward control uses the existing ControlledViaAnd owner and its original
single-bit zero predicate, rather than replacing the source constructor.

The format interface requires injected phase, ordinary-pair and positive-pair
owners. The positive pair's second Program must be the source-selected control
of U's exact inverse, not a guessed primitive reversal. Ecosystem's native
phase adapter retains those explicit ownership boundaries. Unknown children
remain the supplied owners' responsibility. Symbolic physical input, original
serialization failures, mathematical polynomial validation and unavailable
source QEC models remain separate from this constructor's execution scope.
Users continue to bind a Program and append the resulting native operation.

## Closed constructor and direct dependency scope

All three original XPow factories retain their exact surviving fields and
source-selected inverse exponent, rather than replacing XPow(1) by named X.
Original serialization depths 0/1/2, seeded one/two-qubit matrix U, actual
measured ZPow U and CNOT's Toffoli specialization are exercised. Existing
register collections adapt CZPow's two-element q and Toffoli's packed controls
without changing any primitive or measurement-feedback byte. Complete bound
IR and workspace-event receipts remain identical before and after each view.

Stored scalar words and row-major shaped words reuse exact existing ordinary,
inverse and controlled owners. The unchanged serializer can include U as an
additional root at sufficient depth to retain nested CompositeBloq dependencies;
this actual original producer path is tested. A shallow CompositeBloq alone
cannot supply its omitted body. Tagged automatic SDK capture still rejects;
the explicit typed owner supplies that independently accepted controlled body,
without stripping tags or modifying the SDK importer. Missing/invalid owners
fail, while a swapped numerical phase provider produces an identical complete
Program and bound IR without changing the consumer.

The source SDK executes `int(negative_power)`: finite fractional values truncate,
and negative results take forward branches without a surplus inverse loop.
The constructor preserves that literal behavior; the finite component receives
its derived nonnegative execution count. Declared resource fields are a separate
account and are not silently rounded to match execution. A symbolic inverse
power retains a typed bodyless declaration, never evaluates child owners, and
refuses physical binding. Original coefficient/shape serialization failures,
existing mathematical complementarity responsibility and unresolved physical
dimensions retain their previous boundaries.

## Exact calls and numerical evidence

Root calls compare every source-ordered physical operand and predecessor/output
identity, including packed/shaped specialization views. Node numbers are
alpha-renamed in visitation order; source node IDs are not semantic values.
Every actual SU2 invocation is checked against the original theta, phi, lambda
and eps, including precision/(len(P)+1). The owned scalar/Rz/Rx/Rz child chain is
reused unchanged. Complete complex operators include a coherent spectator,
reversed target placement, every small measurement history and zero returned
private work. Normal append/JSON, static state execution and measured trajectory
execution run with source SDK imports blocked. Source-free constructor resolution
uses the existing Simulation phase recursion through the Ecosystem injection.

Scalar U has zero data ports and uses the existing scalar carrier for bare
inverse calls. Its original stored graph retains the three surplus scalar
inverse nodes but renumbers them; it does not lose their phase. The graph's
full tensor is used directly as the numerical reference, while the literal
SDK generator independently retains all three bare scalar inverse calls. The
unchanged SDK unitary protocol nevertheless asserts that a scalar BloqAsCirqGate tensor must
be two-dimensional. Tests retain that refusal and use the unmodified stored
source tensor for the full comparison. Shaped CompositeBloq U likewise retains
a distinct original Circuit.has_unitary decomposition failure; its original
unitary protocol remains an independent executable reference. No original
source is patched and neither view is presented as universal SDK success.

## Physical inventory and declared estimates

Eighteen separate directional input rows are recorded in
`g2-gqsp-component-resources-20261126.json`. Twelve available seven-category
models match unchanged source estimates; six MatrixGate models retain original
no-decomposition failures. Private lease peaks come from actual allocation/free
events, independently of reserved pool capacity and source logical widths.

| Original fixture / child | Inverse power | Actual instructions | Actual measurements | Actual private peak | Declared source model |
| --- | ---: | ---: | ---: | ---: | --- |
| Original XPow degree one | 0 | 17 | 0 | 1 | 2 Clifford, 7 rotations; width 2 |
| Original XPow degree two | 1 | 25 | 0 | 1 | 2 Clifford, 11 rotations; width 2 |
| Original XPow degree two | 5 | 32 | 0 | 1 | 3 Clifford, 11 rotations; width 2 |
| ZPow degree two | 0 / 1 / 5 | 61 / 59 / 60 | 2 | 1 | 2 And, 2 measurements; rotations 11 / 11 / 14 |
| CNOT degree two | 0 / 1 / 5 | 27 / 25 / 26 | 0 | 1 | 2 Toffoli, 9 rotations |
| Matrix U degree two | 0 / 1 / 5 | 27 / 25 / 26 | 0 | 1 | Source QEC model unavailable |
| Scalar U degree two | 0 / 1 / 5 | 27 / 25 / 35 | 0 | 1 | 2 T, 9 rotations; width 1 |

A registerless logical phase has zero declared cost while its existing native
carrier uses actual gates and one private bit. These are different accounting
bases, not a claimed equality between primitive counts and resource estimates.
All leases return; source models are never synthesized from matrix atoms.

## Batch validation and remaining parent scope

The final batch passes 284 concentrated cases without skips in each actual
Cirq 1.7.0/Qiskit 2.5.0 and Cirq 1.6.1/Qiskit 2.0.0 environment, plus 145 local
integration cases. Four files pass strict typing, Black and Ruff. Both team
scopes, English/links, architecture, internal IR and protected API checks pass.
Qualtran 0.7.0 and the real Kickmix 0.1 Simulator are asserted; six complete
original source/test/dependency blobs match the frozen CSV identities. Existing
full-file mathematical, tensor, model and original-test receipts are reused.

The constructor union is now 202/303: 201 Compiler and 40 Ecosystem keys with
39 shared keys. The 101 unmatched entries retain their source roles. Original
Issue860 and degenerate complementary roots remain separate, as do whole-library
reflection/codecs, unspecified child owners and broader original SDK failures.
Q-0654/Q-0655 and Q-0866/Q-0876 are not closed by this constructor delivery.
Pending 118/119/124/126/127 are unchanged. There is no CI wait, scientific
installation, ReadTheDocs, G3 transition or PR merge.

## Logical dependency closure, 2026-11-28

The component now admits typed source leaf declarations without fabricating a
body. Empty modules and physical-context modules remain invalid; required
through-port, signal, inverse and control signature checks remain in place.
Physical binding still rejects reachable missing bodies. Unused declarations do
not prevent binding a genuinely complete reachable branch: the positive inverse
branch is exercised against the full original complex operator, while a branch
that reaches the missing bare inverse is refused.

`test_gqsp_logical_source_closure.py` covers actual source THCRotations U at
Laurent powers 0/1/2/5, available control-wrapper calls and exact SU2 definitions.
The separately declared QEC models are, in category order And/Clifford/rotation/
measurement: 104/160/9/104, 106/161/9/106, 108/162/9/108, and 108/162/9/108.
The last row additionally has 81 Toffoli. All source/native logical peaks are14.
These are models, not measured primitive inventories. Existing 18 executable
primitive/model rows remain authoritative. Seeded three-target matrix cases
extend the same executable constructor path without changing its atom synthesis.

The complete-file review in GENERALIZED_QSP_FILE_AUDIT.md supersedes the earlier
Q-0654/Q-0655 open disposition under the approved faithful-source rule. Whole
serializer parents Q-0866/Q-0876 remain open. Source defects and unspecified
execution owners are retained, not silently repaired or priced as zero gates.
