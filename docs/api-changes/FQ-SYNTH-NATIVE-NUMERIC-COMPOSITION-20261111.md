# Native numeric composition before explicit quantization

Status: next implementation contract; not implemented by the fixed-point
foundation. Native acceptance remains 208/216. Classification:
`single_device_fast_path`.

## Evidence and scope

The shared bounded fixed-point authority is now in Compiler and replaces the
Qmod arithmetic implementation. Its separate Python-AST consumer and existing
Qmod full-state tests pass, with all 216 original rows and six approved revision
rows unchanged. This does not make native numeric programs composable.

Four independent native probes still refuse: initialization of a typed hidden
numeric local, arithmetic reading a fractional register, a helper with numeric
ports, and a non-binary decimal literal. The first three are also necessary for
exact binary arithmetic and must be solved without silently selecting the
Qmod-specific eight-bit approximation profile. The fourth requires a subsequent
explicit quantization admission contract. Neither step alone certifies Brownian.

## Existing user vocabulary and semantic obligations

Reuse `qfunc`, `QNum[width, signed, fractional_bits]`, `Const`, `Output`, typed
root locals, `|=`, `drop`, and `Program.numeric_probabilities`. Add no public
wrapper, alternate program object, decoding API, or implicit precision setting.
A stored word denotes its signed or unsigned integer divided by
`2**fractional_bits`. Raw bitwise XOR retains its existing bit-pattern meaning.

A fresh typed root local may be initialized with exact numeric `|=` just as an
owned Output may. It must obey the existing initialization, whole-register drop,
alias, root storage and no-use-after-drop rules. A dropped numeric register is a
retained quantum environment, not a proved clean work bit. Do not silently clear,
reuse or classify it as scratch. Duplicate initialization must refuse.

Arithmetic may read admitted signed and fractional scalar values. Addition and
subtraction align to the larger scale; multiplication adds scales; powers scale
by their literal degree. Negation, comparisons and admitted extrema interpret
numeric values rather than raw encodings. Infer signed bounds and widths in the
shared exact word authority. Preserve current operator admission and existing
budgets. Never truncate low bits, wrap overflow, round decimals or reinterpret a
fractional register as an integer to make a program fit. Unsupported operators
and non-binary decimals continue to refuse at their source location.

An inferred `Const[QNum]` helper port receives the actual scalar's complete
format; an explicit numeric annotation must agree with the actual format. Fresh
Output formats propagate back to callers. Function specialization and reuse must
key on width, sign and scale, not width alone. Read-only borrowing, disjoint
writable outputs, cross-call aliases, output escape and recursion checks remain
in force. This admission does not grant controlled or inverse numeric helper
semantics without their existing reversibility checks.

## Ownership and reuse

Compiler extends existing exact `WordExpression`, `_Capture`/`_Body`, port
specialization and register verification. Existing qint operations, register
records and lowering remain authoritative. Preserve numeric format metadata
across calls. Reuse existing signed extension and named numeric decoding.
Ecosystem continues consuming the shared exact and bounded arithmetic; it does
not implement native syntax. Services need changes only if a complete inferred
format is lost at its existing program-result boundary. Split implementation
commits by owning domain after this contract.

Do not copy the Qmod lexical planner into a second native compiler. Do not
translate native functions into Qmod text, execute their Python bodies, substitute
integer payoff tables, or prepare their final state directly. The bounded
fixed-point evaluator remains available for the later explicit quantized policy;
exact composition is not permission to select that policy implicitly.

## Acceptance before the quantized Brownian slice

- Exercise typed hidden local initialization and arithmetic reads on positive,
  negative and extreme signed/scaled words. Include coherent complex inputs,
  complete absolute amplitudes, named Fraction labels and actual resources.
- Compose at least two numeric helpers, including inferred and explicitly typed
  ports. Call a helper with equal-width values of different scales and signs to
  catch incorrect specialization-cache reuse. Compare with independent Circuit
  and rational/integer references, without aligning away a global phase.
- Verify dropped local environment and the reduced published-output state;
  verify actual scratch bits return to zero separately. Preserve any correlations
  with the environment rather than merely matching published probabilities.
- Refuse format mismatch, precision loss, overflow, read-before-initialization,
  reinitialization, use after drop, writable overlap, captured aliases and Output
  escape. Check real operation, width, workspace and memory budget refusals.
- Keep existing exact arithmetic regression and non-binary decimal refusal;
  pass strict typing and ownership checks. Replay all 216 unchanged originals
  and the six approved revisions, retaining original resource baselines.

Only then specify and implement native explicit quantization, including the
literal staging needed to reproduce the full Brownian recurrence. Native
Brownian acceptance still requires the original scope, helper composition,
quantized intermediate values, complete retained environment, independent
Circuit full-state/reduced-state checks and fresh corpus replay. The remaining
three finance programs also require their own complete dependency review.
