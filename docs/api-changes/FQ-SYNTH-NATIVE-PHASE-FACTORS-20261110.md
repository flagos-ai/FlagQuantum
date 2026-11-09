# Shared Boolean factors for native real phases

The standing native-library authorization covers this internal, contract-first
replacement. Classification is single_device_fast_path. Implementation is
authorized; release is not. No public export, marker signature, IR operation,
Runtime route, or serialized result schema changes.

## Verified need and ownership

The retained Facility Location program has a squared count of empty facilities.
Its native polynomial expansion currently refuses at the existing 256-monomial
bound. The Qmod adapter already handles this case by preserving products of
three to six distinct signed Boolean literals and sharing them as temporary
predicate values. Raising the expansion bound would duplicate this logic and
conceal the actual missing optimization.

Move that existing factor-selection algorithm into private Compiler ownership.
Keep the source adapter as syntax conversion only, and reuse the same algorithm
from native phase capture. The shared routine receives syntax access/rewrite
callbacks and resolved logical locations, and returns a rewritten expression
plus ordered signed Boolean cubes. Source-specific node construction,
diagnostics and storage emission remain at the frontend boundary. There is one
factor-selection implementation; neither frontend owns a numerical oracle.

## Semantic and resource contract

Only expansion-budget refusal admits factoring. Already-supported small
expressions retain their existing capture path and resource counts. Other
validation failures propagate unchanged. Preserve the existing limits: at most
eight factors, three to six distinct noncontradictory bit literals per factor,
and all existing polynomial degree, term, product and physical-input bounds.
Do not factor signed/scaled numeric words as unsigned bits.

Each predicate is computed into a fresh logical local using the existing exact
word compiler and checked qreg.local/compute operations. Apply the real phase
through existing effects, then uncompute and release locals in reverse order.
Repeated cubes share a predicate within the phase expression. Work bits must
return to zero coherently. Their allocation and physical implementation remain
under existing verification, candidate selection and program budgets.

Algebraic cancellation must not conceal uninitialized or Const/illegal reads.
Deduplication must preserve logical locations across helper views and multiple
owners. Contradictory literals retain the established unfactored behavior.
Absolute constant phase, asymmetric application bit order and the original
finite-angle checks are preserved. No extra controlled-region semantics is
admitted by this replacement; conditional budget refusals remain separate.

## Acceptance

Use a replacement/conformance test behind both unchanged consumers. Replay the
existing Qmod phase-factor tests and frozen original results without editing
source bytes or baseline rows. Independently validate positive/negative signed
literals, reordered/helper views, shared cubes, constant and relative phases,
coherent workspace cleanup, unsupported cases and actual budget refusals.

Preserve the complete Facility Location original, all twelve decisions, all
coefficients, all five cost/mixer layers and the squared count of empty
facilities. Native and original programs must agree with an independent Circuit
in full complex amplitudes for two nonzero parameter sets. The independent
reference evaluates application constraints and finite truth factors without
calling either frontend or loading source from user examples. Other candidate
originals count only after their own complete dependency and amplitude checks.

After each accepted batch, rerun all 216 immutable source cases and the six
separately approved revisions, preserve source/resource evidence, regenerate
native navigation, and update the delivery ledger. Notebook and quantum-program
coverage remain separate. No integration PR or full remote CI before complete
native corpus acceptance.
