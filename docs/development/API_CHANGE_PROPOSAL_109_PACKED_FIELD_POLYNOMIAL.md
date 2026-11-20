# Proposal 109 — Packed binary-field polynomial Program ports

## Review status

Pending explicit user approval. This proposal changes protected Program type
admission. Proposal 108 approves classical integer words and does not approve
this separate quantum polynomial encoding. No implementation is authorized by
this draft.

## Verified requirement

The complete frozen Qualtran v0.7.0 GFPolySplit/GFPolyJoin file uses QGFPoly
ports and QGF coefficient arrays. FlagQuantum already stores the corresponding
metadata as RegisterType with encoding finite_field_polynomial, produced by
FiniteFieldPolynomialEncoding.register_type. Core currently rejects that
encoding in both logical and executable Program validation. Existing Compiler
views therefore use an opaque packed port and retain typed field coefficients.
The reference file is qualtran/bloqs/gf_poly_arithmetic/gf_poly_split_and_join.py
at bcb17bb25360ed68ce3bc084645db3e929799389, blob
966e38d6070cedf8c99f83dd2bd7fc5eb1743854.

## Proposed admission

Admit existing canonical quantum finite_field_polynomial descriptors with
concrete nonnegative polynomial_degree, positive binary coefficient degree,
characteristic 2 and order 2**degree. The quantum width must equal
(polynomial_degree + 1) * coefficient degree. Classical width must be zero.
Parameter order remains polynomial_degree followed by the existing canonical
finite_field parameter order, with its optional complete polynomial identity.
Reject duplicate, missing, reordered, nonintegral and inconsistent metadata.
Reuse existing binary coefficient descriptor validation. Core remains independent
of numerical libraries and foreign SDKs.

Symbolic metadata and nonbinary coefficients keep their separately inventoried
execution boundaries. Classical finite-field and polynomial storage are outside
this proposal. No new public class, opcode, operation field or root export is
introduced.

## Binding and ownership

Use existing Split, Join, Cast, signatures, linear ownership and call binding.
Packed quantum words split into highest-degree-first coefficients with MSB-first
bits, including leading zero coefficients. Join restores the exact declared
polynomial descriptor and width. Views emit zero gates, allocate no work and
create no classical slots. Shaped arrays preserve row-major ordering.

Compiler retains its current opaque-view entry for compatibility and adds an
explicit typed packed route using existing representations. Ordinary users
continue circuit.append(op). Existing field addition, constant addition and
coefficient-view consumers share the same owner. Existing opaque payloads and
CBit/integer contracts remain covered by exact compatibility fixtures.

## Ownership and implementation order

1. Integration records approval, machine contract and compatibility fixture.
2. Core validates canonical polynomial port metadata and existing read/write
   ownership through Program verification.
3. Compiler binds typed packed views and composes the existing coefficient
   operations, preserving source aliases and child order.
4. Simulation and Ecosystem reuse their existing codec and metadata projection
   owners. Foreign method signatures do not become public native APIs.

## Verification

Inventory both full source classes, all methods, two factories and document
specifications. Reuse accepted numerical, source tensor, drawer and opaque-view
evidence. Verify typed scalar and shaped identity, nested calls, split/join
roundtrips, invalid descriptors, nominal field mismatch and linear-handle
misuse. Compare source wire order and outputs, full complex states with
spectators, clean work and gate-free resources. Verify typed polynomial add and
constant-add through normal append with the original child streams. Preserve
source atomic-decomposition refusals and generic serialization boundaries.
Run focused regression, local integration, strict typing, formatting, scope,
architecture and IR/API checks. Update progress only after acceptance.
