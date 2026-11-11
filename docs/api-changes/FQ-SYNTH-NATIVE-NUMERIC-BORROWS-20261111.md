# Checked native unsigned numeric borrows

The user authorized completing all 216 native quantum programs and filling the
numeric-view capability gap. This unreleased extension reuses existing typed
qfunc helpers, static locations, indexed reversible gates and unsigned word
expressions. No public symbol/signature, serialized opcode, Stable Core contract
or execution backend changes. Release authorization remains false.

A plain QNum helper can borrow a static QNum slice or an explicit list/tuple of
one to eight existing scalar/register/view components. Components are flattened
in logical least-significant-first order, matching existing native matrix views
and original Qmod binding order. The physical locations must be nonempty,
distinct, initialized and writable; formal widths must agree. This is an alias,
never allocation, reinitialization, copying or ownership transfer. A bare QArray
still cannot masquerade as a whole numeric register. Signed/fractional storage
cannot be reinterpreted by this raw unsigned path.

Raw +=/-= applies modulo 2**view_width, including carry across components and
wraparound. It touches only selected locations and retains the existing bounded
static offset, call/AST/loop and physical resource limits. Controlled updates
must be disjoint from every dependency register of their condition. Adjoint
and within/apply reuse checked indexed effects and preserve absolute phase.

Unsigned expression reads in borrowed helpers resolve each formal word using
its ordered actual physical bits before invoking the existing word compiler.
Static subviews are relative to the formal view, not the parent register.
Initialization and Const checks remain authoritative. Input/output alias,
read/write borrowing, ownership and Output escape refusals remain conservative
at the existing register boundary; this extension does not authorize writing an
unselected bit of a register used as a numeric predicate dependency. Signed or
fixed-point sliced arithmetic and runtime/dynamic views remain unsupported.

In-place real/probability preparation may use the same ordered partial or
composed initialized writable storage. It applies the existing full unitary on
every input column, not an assumed fresh-zero map. Nested controls prepend
checked scalar locations to the existing indexed preparation effect; combined
control/target locations remain bounded at eight, preserving inverse and branch
global phase. Const, overlap and ownership rules are unchanged. No new opcode,
control-state schema or resource ceiling is introduced.

## Acceptance

- Compare all small basis words, nonzero surrounding bits and coherent complex
  states to an independent Circuit permutation, including arbitrary component
  order, partial QNum views, nested helpers, wraparound and clean work.
- Check controlled increment/decrement, adjoint interference and within/apply
  with unchanged global phase and exact input read/write effects.
- Check unsigned sliced helper predicates against every basis value and an
  independent coherent Circuit; no temporary source copy or fresh-zero premise.
- Refuse duplicate/overlapping components, wrong widths, Const/uninitialized
  storage, condition dependencies, signed/fractional reinterpretation, dynamic
  or invalid slices, Output escape and actual gate/work/width budgets.
- Re-run the full 216-source and six approved-revision corpus locally, preserving
  immutable original identities and resource baselines. Count native and complete
  Notebook workflows separately. Do not open integration CI before native216.
