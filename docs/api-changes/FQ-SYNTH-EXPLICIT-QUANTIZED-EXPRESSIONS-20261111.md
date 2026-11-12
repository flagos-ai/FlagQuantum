# Explicit quantized expressions for native qfunc

Status: implemented locally on the native-example integration branch; acceptance
and coverage evidence are tracked in the catalog and delivery ledger. The
contract was admitted at 208/216 native cases. Classification:
`single_device_fast_path`.

## Why a separate explicit marker is required

Native numeric composition now reuses exact signed/scaled `WordExpression`,
`QNum` metadata, and existing qfunc specialization. It intentionally continues
to refuse non-binary decimal `0.1`. The bounded Qmod numeric foundation preserves
a distinct source profile: decimal literals are truncated toward zero at a
20-bit binary staging scale, then quantized down to eight fractional bits;
products and powers truncate down at their admitted result scale. Ordinary
`QNum` arithmetic must not silently acquire that approximation.

The original Brownian Chebyshev program has 23 physical bits but only five
independent prepared selector bits. Its complete recurrence fits the existing
32-row bounded table: 15-bit signed `G`, eight fractional bits, 32 branches,
24 output labels, and two clean work bits. An independent Fraction recurrence
already matches all original 23-bit amplitudes and absolute phases within
5.1e-15. This proves the oracle and boundary; it is not a native acceptance.

## User entry and profile

Add one captured pure-expression marker:

```python
score |= quantize(
    -6.283185307179586 * (t - 0.5),
    fractional_bits=8,
    literal_bits=20,
)
```

`quantize` is a `flagquantum.synth` public symbol that cannot execute as a host
function. Python capture recognizes its exact resolved symbol and retained AST;
never evaluate the wrapped expression or call arbitrary host code. It may appear
only as the complete right-hand side of fresh `|=` numeric initialization.
In-place XOR is a separate existing profile and is not extended by this marker. It is not a general nested host function, classical
parameter transformation, phase function, gate angle or Python runtime helper.

The required `fractional_bits` and `literal_bits` are non-Boolean integer
literals. The first implementation admits exactly `(8, 20)`, matching the
already verified bounded arithmetic authority; other profiles refuse rather than
pretending to be implemented. There is no default precision, rounding selector
or silent fallback. `fractional_bits` caps intermediate precision rather than
forcing every word to store eight fractional bits. The result uses its inferred
binary scale, which can be smaller for exact dyadic values; assignment metadata
and output `QNum` format must agree. Infer signedness, bounds and width from the
same scalar arithmetic authority. Existing exact `QNum` operators and exact binary literal behavior
remain unchanged outside the marker. The existing fixed-point XOR profile and
all its alias, initialization, bounds and nonbinary-coefficient refusals remain
unchanged unless separately and specifically extended by this contract.

## Quantization semantics

- A real literal first truncates toward zero after multiplying by `2**literal_bits`.
- The staged literal is then floored to the expression's requested fractional scale.
- Addition and subtraction align exact binary scales and do not round.
- Multiplication and literal powers floor toward negative infinity at the
  admitted result scale `min(sum of operand scales, fractional_bits)`.
- Negation and bounded powers preserve signed two's-complement interpretation.
  Powers accept source-independent literal degrees 1 through 4.
- Output values are never wrapped or truncated to fit a declaration. Inferred
  width must contain the full signed/unsigned range; an explicit declaration must
  match it. Overflow, scale disagreement, unsupported operators and
  non-finite/out-of-range literals fail at the original source location.
- Each visible quantum input is interpreted with its actual `QNum` sign and
  fractional format. Helpers receive and return complete width/sign/scale
  formats; specialization identity contains all three. Reads are Const-only;
  writable outputs are disjoint, and controlled, adjoint, recursive, aliasing or
  escaping numeric helpers refuse. A fresh owned caller local may receive a helper
  Output and remains owned by that caller; this is initialization, not escape.
  This narrowly supersedes the blanket hidden-Output refusal for the closed
  quantized profile. Initialized locals cannot be passed as fresh Outputs;
  Const storage cannot be published, captured or returned as a new owned handle.
- The first program profile is closed, acyclic, bounded and precision annotated.
  Its independent prepared binary selectors span at most 32 branches, with every
  coherent branch represented. Source types alone do not establish this bound.
  A 15-bit derived word can be admitted only after proving that its values are
  functions of those same selectors; it is not treated as five unconstrained
  read bits. Track a shared selector domain through assignments and helper
  bindings, preserving correlations. A helper called on independently prepared
  wider input must refuse if its proven joint support exceeds 32 rows.
- Do not infer support from simulation samples or drop rows with small amplitude.
  Only statically checked preparations and exact data flow establish the closed
  domain. Retain read witnesses for cancelled expressions and unreachable-word
  assumptions in the proof. Phase/preparation effects must remain actual quantum
  operations, including any admitted complex phases.
- A compressed environment is valid only after the values explicitly dropped by
  the source distinguish every prepared branch. Otherwise refuse or retain the
  full environment under a separately verified implementation. Check this
  property using actual value labels, not source variable names or row IDs.
- Exceeding branch, AST, helper depth, statement, width, gate, workspace or memory
  limits refuses at source. Reuse the existing 32-row/twenty-bit bounded arithmetic
  limits and verify the complete resulting instruction sequence. Broader support
  requires measured evidence, not an unbounded enumeration fallback.

## Lowering and ownership

Compiler owns the expression callback adapter and table lowering. Reuse
`compiler/_construction/fixed_point.py` for scalar arithmetic and
`compiler/_fixed_word.truth_cubes` / existing `qreg.control_x` effects for
reversible truth-table writes. Reuse existing QNum registers, output formats,
helper calls, truth-table gate verification, resource accounting and readout.
Do not add an opcode, numeric IR, simulator shortcut, backend-specific gate,
source importer path or alternate result decoder. The finite scalar/selector
facts are compile-time inference data, not a second executable IR. Reuse native
source resolution, port-format specialization and borrow validation. Factor
shared bounded numeric planning/emission out of existing authority where needed;
do not copy the Qmod lexical planner into a parallel native compiler. Qmod syntax
objects must remain outside Compiler. Ecosystem retains Qmod source
recognition, profile selection, helper/ownership planning and current
source-located diagnostics; it delegates scalar arithmetic to the shared
Compiler evaluator. No native example may load a Qmod file.

Owned outputs and hidden locals use current initialization/drop contracts. A
dropped register remains physical garbage unless independently proved zero; it
must not be counted as workspace or silently erased. Independent acceptance
compares every physical amplitude, absolute phase and reduced published-output
state. Scratch cleanliness is checked separately. Original input superposition,
all 23 physical bits, hidden branch correlations and source output width remain.
No precomputed final-state preparation, integer-only payoff, discarded-branch
loss or width-reduced surrogate can count as the original.

## Brownian acceptance gate

Implement the original 23-bit recurrence with named `@qfunc` helpers using the
explicit marker on each precision-bearing expression. Exercise all 32 five-bit
branches and all three nonzero Chebyshev coefficients, including signed extrema.
Compare the complete native statevector and reduced named `(t, G)` density/state
against a separately implemented Circuit oracle, with no phase alignment. Verify
the 15-bit signed/eight-fraction format, 24 output labels, exact branch support,
both work bits zero, the proved retained branch environment, 23-bit physical scope, and
actual width/operation/depth/workspace/memory budgets. Also test coherent
superpositions, existing nonzero XOR updates as an unchanged regression, positive/negative
rounding boundaries and unsupported over-budget input. Re-run existing exact
numeric, fixed-XOR, quantized Qmod and all helper/ownership regressions. Refresh
all 216 original sources and six approved revisions; preserve every source hash,
outcome and resource row. Update catalog, immutable evidence and delivery ledger.

This batch may credit only Brownian when the original itself passes. Then audit
autocallable and both rainbow originals at their exact call sites; quantization
availability alone does not establish those cases. The other four pending
programs keep their independent blocker chains. Do not open an integration PR or
run full remote CI before all 216 native acceptances pass.
