# Modular declaration whole-file audit

The complete reference file is `bloqs/arithmetic/_shims.py` from Qualtran v0.7.0
at `bcb17bb25360ed68ce3bc084645db3e929799389`. It defines four immutable
resource declarations, their cached signatures and direct call graphs. Its
direct dependencies are CAdd, BitwiseNot, Toffoli and TwoBitCSwap.

Compiler `registers/modular/modular_declarations.py` owns their native kind-based
interface. `modular_arithmetic_signature` returns existing Core Signature and
Register values. MultiCToffoli's control is an array of bits with shape `(n,)`.
CSub and Lt use unsigned x/y words. CHalf uses an unsigned x word. Every port
is through, signatures retain source order and all words use existing MSB-first
encoding. These are metadata declarations with no work-bit ownership.

The existing flat-layout query now reads this same signature, retaining the
original width object for each port. Its existing admission checks remain at
that query. Raw typed signature inspection preserves negative diagnostic and
symbolic widths. Physical execution and recursive models keep their existing
admission policies. The source `signed` field changes Lt identity without
changing its dtype or direct calls.

Direct calls, source insertion order, symbolic multiplicities, independent
category profiles and peak fallback reuse ARITHMETIC_RESOURCE_GRAPH.md and
REGISTER_MODULAR_DECLARATIONS.md. CSub retains Montgomery CAdd and three
Montgomery BitwiseNot calls. Adjoint declarations use the existing graph owner.
The source has no executable or classical decomposition for these declarations.
Gate streams, complex amplitudes and measurement branches therefore terminate
at the actual CAdd and fixed-gate dependencies, whose existing evidence is
reused. No executable operation is inferred from a cost declaration.

## Acceptance and retained boundaries

Typed signatures match all four original classes at negative, zero, concrete
and symbolic widths, including Lt's two signed-field values. Checks include
exact dtype, array shape, direction, scalar and total widths, symbol identity,
and the old flat query's use of the new authoritative signature. Existing
flat-query negative-width refusal is preserved. No Core contract or stable
public export changes.

The concentrated batch passed 1187 cases with two paired source/native NumPy
overflow warnings from existing controlled-addition tests. Formatting, lint,
strict typing of both changed Python files, Compiler scope, architecture,
multilevel IR and protected API checks passed. The existing
`python -m examples.algorithms.modular_resource_declarations` example passed.
The source file blob is asserted without reference modifications. Logs are
retained as `ftqc-reference-env/g2-modular-declaration-file-*.log`.

Q-0172's complete declaration responsibility is accepted. Source diagnostic
negative multiplicities, zero-width CSub QEC/width failures and source absence
of executable/classical decompositions remain explicit. The independent
three-bit models report one Toffoli/peak4, CSub five And plus 23 Clifford and
five measurements/peak12, Lt three Toffoli/peak7, and CHalf three controlled
swaps/peak4. These are modeled declarations. They are separate from the
executed CAdd dependency's primitive and all-small-measurement-history evidence.
