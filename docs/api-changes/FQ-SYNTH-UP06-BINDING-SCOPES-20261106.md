Next candidate: checked binding-only within/apply for the unchanged quantum_autoencoder.qmod.

Evidence before implementation:
- quantum_autoencoder.qmod first refuses on line 21 {coded, trash} -> x inside within.
- Its apply body contains two bounded nested repeat blocks with RY and CX only.
- The bound encoder helper has an unsized local qbit[] alias declaration.
- angle_encoding takes {coded, trash} as one helper array actual; current array_view
  handles brace items as scalar locations only, while bind_registers already flattens
  whole source registers. Introduce one consistent recursive concatenation view.
- Finite real arrays, RY(pi*input), static length arithmetic, CX, hidden data state,
  bounded helper specialization and physical lowering already exist.
- quantum_counting.qmod and qmci_pi_estimation.qmod also need binding scopes, but
  still need amplitude_amplification and checked record reinterpretation helpers;
  do not claim direct unlock or original execution for those two.
- General within/apply quantum inverse composition, classical loops in other
  quantum scopes, signed/fractional views and ownership-changing apply blocks stay
  outside this slice. Existing Hadamard and arithmetic within paths stay frozen.

Proposed boundary:
- A within preparation made entirely of -> bindings may create a temporary local
  view. No allocation or numerical inverse is implied by this metadata scope.
- Initialize and disjoint-check source storage through existing binding checks.
- On exit (and on refusal), restore suspended names, views and width metadata to
  the incoming state; the temporary alias cannot escape or own output storage.
- Admit bounded static gate/helper loops in apply for this binding-only scope.
  Reject allocating, initializing, releasing, dropping or returning fresh storage
  by validating the actual specialized owned effects; keep all expansion caps.
- Declare a local unsized qbit[] alias in a bound helper without allocating it.
  Using it before binding or after lexical restoration must refuse.
- Concatenated whole arrays and scalar/static indexed locations preserve existing
  low-to-high source bit order and helper alias checks.
- Extract scoped binding parse/lowering into a private domain module so syntax.py
  remains <=1250 lines; do not create new public concepts or policies.

Acceptance:
- Copy unchanged original with pinned source/license manifest.
- Bind two distinct nonzero 10-weight and 4-data arrays. Independent public
  Circuit prepares angles pi*data in low-to-high concatenated coded/trash order,
  then two RY/CX layers and final trash rotations. Compare full complex128 vector,
  absolute global phase, named trash marginals and all actual scratch-zero slices.
- Test temporary alias lifetime, restore original inputs, nested binding scopes,
  slice/concatenation order, partial and repeated storage overlap, uninitialized
  inputs, Const mutation, fresh Output escape, width mismatch, zero/negative and
  large loop bounds, out-of-range angle index and physical resource caps.
- Native qfunc example uses supported scalar outputs, static real captures and
  explicit gate composition, without claiming new native array/loop support.
- Full 216 audit must preserve all previous 135 resource rows and source blobs.
- Keep both known invalid original inputs as source-semantic refusals; no original
  or historical measurement baseline is rewritten. No integration PR/full CI yet.

Nested static loops must reject reuse of an active loop-index name, preserving
the outer static environment. Negative bounds may refuse during parsing before
the 0..256 semantic bound check. Existing effect expansion limits still apply
when loop products exceed 4096 operations.
