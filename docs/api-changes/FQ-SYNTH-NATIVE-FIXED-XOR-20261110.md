# Native bounded fixed-point expression XOR

Status: dependency audit and observed native refusal; implementation pending.

## Requirement and reuse

The unchanged tanh program prepares a four-bit unsigned all-fractional input,
selects one of four dyadic linear models by a copied MSB label, and XORs the
quantized value into an allocated four-bit fractional output. Native fractional
expression XOR currently refuses before lowering. Fresh numeric assignment
cannot substitute for a reversible update of arbitrary nonzero target storage.

The Qmod adapter already owns bounded fixed XOR syntax and uses exact rational
expression evaluation, significance alignment, floor-and-mask quantization and
existing checked truth-XOR effects. Extract the source-neutral numerical
semantics into Compiler rather than duplicating them or importing Ecosystem into
Compiler. Keep Qmod parsing, source diagnostics and static scope in Ecosystem.
Native capture resolves words, scales, dyadic constants and static guards, then
uses the same Compiler implementation. No public API or numerical runtime is
added; the existing `^=` syntax gains a checked fractional profile.

## Contract

The target is a whole initialized writable unsigned fixed-point scalar, at most
eight bits. Inputs and any static equality selector occupy at most seven distinct
read bits and are disjoint from the target. Exact dyadic constants are bounded
before rational arithmetic. Admit only explicitly checked pure operations;
unsupported operators, non-dyadic coefficients, oversized syntax/intermediates,
signed input profiles and invalid selector ranges refuse before circuit emission.

Compute the exact value, multiply by the destination binary scale, floor toward
negative infinity, then mask to the target width. XOR these raw bits into the
existing target. This precision loss is the explicit fixed-update contract;
fresh exact assignment continues to reject discarded precision. Repeating the
update is its inverse for every target value. Constant cancellation and zero
updates retain initialization, alias and format witnesses. All scratch must be
uncomputed; copied selector views are read-only, not fresh quantum information.
Normal helper ownership, Const restrictions, static array binding, output escape
and resource ceilings remain authoritative. Do not relax unrelated callback or
view boundaries to make the example pass.

## Acceptance and split

1. Extract the existing exact fixed-word XOR semantics into Compiler with
   source-neutral inputs and exhaustive word/quantization tests; keep the Qmod
   adapter using that authority and retain its existing refusals.
2. Add native whole-word update and checked static selector/helper capture;
   preserve real-array binding and helper actual/formal aliases. Prove nonzero
   XOR targets, repeated updates, signed complex superpositions, clean work,
   boundaries and format/alias/initialization/budget refusals.
3. Reproduce the original tanh program with all four coefficient pairs and all
   sixteen coherent inputs. Independently calculate rational quantized results
   and construct a Circuit reference; compare complete absolute amplitudes and
   every work amplitude. Preserve copied-label significance and original source
   bytes. Do not use a case-specific hardcoded gate sequence or lookup table.
4. Run all216 originals and six approved revisions after final production edits,
   preserve original raw baselines and explicitly record changed resources.
   Update catalog, all-five-category inventory and immutable delivery evidence
   only after independent acceptance. Broader signed/Brownian/payoff use remains
   a separate dependency review, not implicit coverage.

Starting acceptance is191/216 quantum programs,24/213 Notebook workflows and
14/19 revision-nine merged scopes. No push, integration PR or remote full CI
before complete native216 acceptance.

## Native selection spelling

The native fixed-word expression admits a bounded Python conditional expression
`yes if comparison else no` with exact rational comparisons, inside the existing
`^=` update. Both branches are captured and validated without host execution;
the chosen exact value is quantized only at the destination. This expresses a
piecewise computation directly and avoids manufacturing a copied quantum label.
For the original four equal-width segments, comparisons against1/4,1/2 and3/4
are precisely equivalent to reading the two most significant raw input bits.
No ordinary Python `if` is reinterpreted as quantum control. Conditional helper
callbacks and mutable numeric helper bodies remain restricted: the admitted
numeric borrowing profile contains exactly one fixed-word XOR update.

The captured `QNum` annotation methods accept `float` alongside `QNum` and
`int` for ordered comparisons and in-place XOR. This typing expansion describes
the checked dyadic expression profile above; it does not enable host execution
or relax the existing raw integer-mask rules. No exports or call ordering change.
