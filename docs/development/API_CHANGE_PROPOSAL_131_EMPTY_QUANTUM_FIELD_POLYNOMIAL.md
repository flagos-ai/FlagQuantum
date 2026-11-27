# Proposal 131: empty quantum field-polynomial storage and views

Status: proposed; not approved. No protected implementation has been changed.

## Verified source success and native gap

Locked Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389` permits
`QGFPoly(-1, QGF(p, m))` as an empty coefficient tuple with zero packed qubits.
The actual GFPolySplit/GFPolyJoin SDK hooks succeed for fields GF(2), GF(8),
GF(3), GF(5) and GF(9). Ten source views emit no operation, return zero wires
and preserve the exact two-by-two identity on an idle host qubit. The source
methods are called directly, without modifying or repairing reference code.

Evidence: `ftqc-reference-env/g2-polynomial-empty-sdk-20261127.json` and its
log. The preceding constructor/tensor probe is
`g2-polynomial-empty-intake-20261127.log`.

Native `FieldPolynomialLayout` rejects degree -1, and Core's polynomial
validation requires a nonnegative polynomial degree and positive field storage
width. Proposal 109 introduced that explicit restriction; Proposal 125 did not
change it. Thus current native typed storage cannot represent these successful
source view calls. Using an opaque zero-width word would lose the field and
polynomial metadata and is not the proposed solution.

The source GF2PolyAdd empty logical decomposition also succeeds, but its generic
SDK wrapper and tensor contraction fail with an empty-sequence unpacking error.
GF2PolyAddK rejects every actual constant polynomial at degree -1. Those source
failures remain distinct from the successful atomic views. Degree below -1 has
invalid negative dimensions; an accidental reshape inference is not a valid
negative-width quantum register and will not be admitted.

## Proposed contract

* Permit exactly polynomial degree -1 with zero quantum width, zero classical
  width and an existing canonical quantum coefficient-field descriptor. Keep
  the descriptor and polynomial degree in logical records and serialization.
  Logical symbolic coefficient fields retain their existing order/degree/width
  requirements; multiplying the empty extent by a symbolic coefficient width
  gives exact zero. Known-invalid coefficient metadata remains rejected even
  though the coefficient tuple is empty.
* Admit raw quantum zero-width storage, ownership and Split/Join views through
  the existing Core Program operations and `circuit.append(op)`. No new opcode,
  register kind, public entry, schema field, source-name namespace or backend is
  introduced. The empty shaped coefficient port has extent zero and remains
  typed; it is not flattened into an untyped placeholder.
* Preserve the existing requirement for an actual host wire in physical
  CircuitIR. Zero data is not permission to create a zero-wire physical circuit.
  An idle host wire remains unchanged and is listed separately from the logical
  data count, workspace and source model.
* Admit logical polynomial-addition graphs where the unchanged source supplies
  them. Preserve source SDK/primitive-body failures separately. Do not claim
  numerical polynomial arithmetic or source SDK parity from raw storage.
* Concrete view execution retains the coefficient-width requirements imposed by
  the original SDK hook. Unknown coefficient width may remain a logical typed
  declaration; source-refused physical views are not silently executed.
* Keep degree below -1 rejected. Keep degree -1 with nonzero packed width
  rejected. Do not expand finite-field scalar zero-width storage, classical
  field/polynomial storage, classical reads/writes, constant-addition validation,
  numerical codecs or unknown-width arithmetic.

## Ownership and implementation order

Integration records explicit approval before protected changes. Core then
updates only the quantum field-polynomial validation and zero-extent width
relation; the classical caller retains its existing positive-width field rule.
Compiler reuses the existing coefficient layout, rebindings and shared
polynomial call schedule. Simulation value codecs keep their original success
and failure behavior. Examples compose existing owners; Compiler executes no
numerical kernel and Core imports no implementation layer.

## Required acceptance

1. Reproduce all ten unchanged source view successes and the named source
   logical/SDK/tensor/constant-addition failures. Retain frozen source hashes.
2. Verify neutral descriptor metadata, empty array side/shape, typed call
   serialization roles and both inverse view directions. Reject malformed
   metadata, nonzero packed widths, degrees below -1, classical zero-field
   storage and source-symbolic-degree constructor failures.
3. Bind both views with an idle host wire. Assert zero primitive gates, no
   measurement, reset, allocation, feedback or workspace effects, and unchanged
   complete complex amplitudes including entangled host spectators.
4. Exercise both view directions and round trips through normal append. Replace
   an implementation/provider at the existing boundary without modifying its
   consumer. Preserve all current positive-degree tests.
5. Run strict typing, format, scope, language/links, architecture, IR and local
   integration checks. Update the source inventory and Markdown stage overview
   only after actual acceptance. Keep measured primitive/host counts and source
   declared models separate.

## Why explicit approval is required

Root AGENTS rule 8 requires an approved proposal before protected behavior or
IR changes. Core/AGENTS requires an integration contract before protected type
changes. This extends the accepted parameter range of an existing serialized
register kind and changes its logical/physical storage admission. Existing
Proposals 109/125/126/130 do not authorize polynomial degree -1. Approval of this
proposal will not by itself close Q-0527 or Q-0533.
