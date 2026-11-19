# Native scalar gates on exact numeric encoded bits

Classification: `single_device_fast_path`. This is an unreleased compatible
Compiler implementation extension under the user's standing explicit request
to complete native examples and fill their capability gaps. No Stable Core
symbol, signature, enum, public schema, runtime behavior or dependency changes.
No protected snapshot regeneration or release authorization is requested.

## Concrete blocked workflow

The original flexible-QPE Notebook estimates on two five-bit all-fractional QNum
counters and performs a controlled swap test between their encoded bits. The
existing scalar gate resolver rejects signed or fractional numeric indexing,
even though the indexed qreg gate representation already retains the full
register's verified format. Whole-register numeric QPE requires the fractional
format, so replacing the counters with unsigned arrays or numbers is invalid.

## Compatible implementation

Keep general `locations` and numerical helper borrowing conservative. Only the
private scalar gate target path can resolve static encoded-bit indices on an
already declared signed/fractional QNum. Existing one-bit width, original type,
bounds, initialization, Const, alias, helper, operation and resource checks still
apply. Recursive static slices inside a scalar gate preserve their relative
location; the ultimate target must be exactly one bit. No sign or scale metadata
is rewritten. Scalar bits are storage locations, not an inferred unsigned word.

Unsigned arithmetic borrowing continues to refuse signed/fractional source
slices. This neither implements sliced fixed-point arithmetic nor permits
format reinterpretation. No new matrix opcode, allocation mode or API is needed.
SWAP's controlled helper is expressed using existing three-CX decomposition;
it does not expand the general controlled-operation admission.

## Acceptance

- Every encoded three-bit input for signed integer, signed fractional and
  unsigned all-fractional formats matches an independent Circuit for RZ/CX/SWAP.
- Coherent control, absolute phase and adjoint/helper composition agree.
- Out-of-range, bool/dynamic index, multi-bit scalar, overlap, Const and
  uninitialized effects fail. Signed/fractional arithmetic borrows still fail.
- Both original-size seven/thirteen-bit flexible-QPE circuits, original ordered
  Hamiltonian and fresh eigenvector, per-power repetitions, sequential shared
  data and full retained environment agree with independent Circuit states.
- Preserve 216 frozen original identities and approved alternatives. Run focused
  regression, strict typing, format/lint, public contracts, architecture/docs,
  source provenance, final Notebook/CLI and frozen local default checks.

## Diagnostic evidence

An external copied package, never the frozen delivery checkout, produced maximum
absolute errors 1.1125e-14 and 2.0872e-14. The exact finite-QPE overlap statistic
is 0.9570620128584528; seeded 1024-draw observation is 0.9453125. The source stored
0.9609375 is historical sampling, not a value to fit. An initial pytest invocation
loaded the unchanged repository package via pytest path precedence and correctly
refused the new scalar views; a preloaded isolated package distinguishes that
red baseline from the candidate regression. Preserve both logs. No diagnostic
result is formal delivery acceptance before installed-source tests finish.
