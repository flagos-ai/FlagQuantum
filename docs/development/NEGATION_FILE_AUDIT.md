# Register negation whole-file audit

Q-0210, Q-0211 and Q-0212 cover the negation notebook, implementation and
tests at Qualtran v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389`.
All three original Git blobs match the frozen inventory.

## Ownership and contracts

Compiler owns big-endian wire binding, complement, increment and feedback.
Simulation owns the original classical composition and its intermediate
validation. The target is THRU. A width-n word is mapped to its additive
inverse modulo 2**n. The caller supplies n clean constant wires, n-1 clean
carry wires and n-1 feedback slots. Every quantum work wire returns to
zero. The feedback slots hold the latest measurement outcomes. Negation
is an involution.

The existing `negate_ir` executes BitwiseNot followed by AddK(1). The
single-control addition owner now also composes controlled negation from
the shared controlled complement and controlled AddK(1). The increment
controls the constant load and unload. The adder itself runs on both
branches. Control-on-zero retains the source wrapper's two outer X gates.
The control and target stay on their original physical wires. No extra
control work is allocated.

| Locked definition | Native disposition |
|---|---|
| Negate dtype and cached signature | Preserve x and both signed factory widths in neutral metadata. The established compiler accepts concrete signed, unsigned and Montgomery bit patterns. |
| Composite build | Existing plain compiler chain and Simulation callback reuse complement and AddK. The original operation order and x binding also match at symbolic width. |
| Generic control of Negate | Both concrete single-control polarities follow the locked wrapper and child order. Concrete grouped controls are covered by the native graph batch below. Symbolic primitive lowering retains its admission boundary. |
| Two factories and documentation specification | Both signatures and complete one-level graphs match. The asset preserves the original topics and adds coherent negative-control execution through ordinary `circuit.append(op)`. |

The source's one-bit signed AddK leaf cannot load +1. The existing source
failure witness stays in the tests alongside the distinct valid native
one-bit permutation. The source zero-width classical failure and unresolved
symbolic classical failures also remain explicit. The implementation's
integer formats and modulo output are described alongside the original
teaching prose.

## Original tests and native execution

The two original test functions cover both factory autotesters and all
unsigned inputs at widths 1, 2, 3, 4 and 8 with signed decoding. The run
reports 9 passed and 1 skipped. The skipped count-mode fixture reports
that Negate only has counts from decomposition. Native declarations compare
both decompositions and their operand/output connections directly.

Prior plain-negation tests cover three encodings, the signed one-bit
failure, per-wire gate order, coherent complex amplitudes with a spectator,
and complete small measurement histories. New controlled tests cover both
polarities at 1/2/3/5/8 bits against the untouched unsigned source
decomposition. They compare emitted primitive identity, per-wire order,
operands, measurement feedback and output maps. At widths 1–3, exact
postselection checks every measurement history on a superposition of
all inputs, controls and spectator values. The expected full states check
clean work and involution separately for each history.

| Eight-bit account | Actual primitive subset and work | Independent source model |
|---|---|---|
| Plain | 81 CX, 28 T/TDG, 7 measurements, 15 clean work wires | 7 And and 7 measurements |
| Control on one | 91 CX, 28 T/TDG, 7 measurements, 15 clean work wires, 0 X | 7 And, 56 Clifford, 7 measurements, 24 logical qubits |
| Control on zero | 91 CX, 28 T/TDG, 7 measurements, 15 clean work wires, 2 X | 7 And, 58 Clifford, 7 measurements, 24 logical qubits |

The controlled model is the shared controlled increment model plus n
logical complement CX gates. Primitive counts include explicit expanded
And and erasure work. Model Clifford categories are kept separately.

## Teaching asset and remaining boundaries

The unmodified 11-cell notebook and the native 13-cell notebook execute in
independent kernels. The native kernel blocks reference imports. The native
asset retains both original instances, the graphical signatures and the
concrete one-level graph. It adds an executable coherent control example.
Q-0210 closes. Q-0211 and Q-0212 retain the original signed one-bit gate
failure, generic symbolic/foreign execution and the original skipped
count-mode boundary.

## Native component graph interface and acceptance

Compiler arithmetic composes the existing typed BitwiseNot and AddK owners
in `negation_component.py`. The input/output is one MSB-first integer THRU word.
Constant and Add carry workspace belongs to AddK and returns clean. Optional
single-bit control preserves its through wire and affects complement and constant
loading. Zero control keeps one outer X/positive-body/X wrapper. The explicit
adjoint reverses the source child calls and uses the existing exact AddK-adjoint;
this graph is kept distinct from merely executing the forward involution twice.
Simulation and dtype value encoding retain their existing ownership.

The native encoded one-bit signed permutation remains executable, while the
original signed AddK(+1) fails its value encoding. Its source failure is recorded
separately from successful unsigned/source and native signed physical evidence.
No modified reference, symbolic physical claim or new numerical kernel is used.


The exact adjoint is explicit AddK-adjoint followed by BitwiseNot. The scalar
control cases and grouped QBit arrays, mixed integer/QBit registers and negative
signed activation values retain source hierarchy, operand connections and
measurement feedback. Unknown two-bit activation retains an unresolved source
conjunction and refuses physical binding. Unknown three-bit activation follows
the source HasLength-to-positive-MultiAnd conversion. This is recorded as source
behavior, with the shared predicate owner's evidence reused.

The final native batch passes 123 focused and 4269 concentrated checks without
skips. Ten pre-existing paired NumPy overflow warnings and two notebook/environment
warnings remain visible. Strict typing for all three touched Python files,
Black, Ruff, compiler scope, architecture, IR and public API checks pass. Five
local integration cases pass, including grouped coherent control and its exact
inverse through ordinary append. Small full source tensors, entangled spectator,
every raw measurement branch and every inverse history pair preserve full complex
amplitudes and zero constant/carry/predicate work. The actual eight-bit source
factory passes all five existing example checks.

For two data bits and a QBit[2] control active at (0, 1), the emitted circuit has
19 CX, 4 X, 8 T/TDG, 2 measurements and 2 resets. The independent source model
has 2 And, 9 Clifford, 2 measurements and a peak of 8 logical qubits. The complete
forward/adjoint, plain/scalar and grouped resource tables are stored in
`g2-negation-native-resources.json` outside the repository with the verification
receipts. Primitive counts and source model categories remain separate.

The default suite reaches 434 passes and 51 skips before the established
`test_every_module_with_examples_is_covered` failure. Its nine reported modules
and the failed test are byte-identical to base 9d16ee53a. Dependency policy has
688 whole-repository findings and zero findings in the touched batch. Frozen
source blobs, Qualtran 0.7.0 and Kickmix 0.1 Simulator are verified. Receipt prefix
`g2-negation-native-*` records this batch. Q-0211/Q-0212 remain open for the
original signed one-bit failure, symbolic physical/generic foreign admission
and original skipped count-mode boundary. Proposal 108 remains pending.

## Complete foreign-consumer disposition after approved contracts

The final consumer audit closes the complete implementation and original test
files using the previously accepted native graph, exact adjoint, classical
callback, notebook and resource evidence. Both actual factory objects preserve
full library bytes at depths zero, one and two and reconstruct equally in the
original. Explicit factories resolve concrete signed execution and symbolic
metadata through existing owners. Physical symbolic widths retain the finite
execution rejection. Both scalar control polarities and integer formats resolve
their original wrapper references to the existing specialized native graph.

Plain original SDK operations at widths 1, 2, 3 and 5 preserve integer metadata,
full primitive/operand/feedback streams and output aliases. Widths through three
also pass full complex superpositions with spectators, every raw measurement
history and clean-work checks through `circuit.append(op)`. Original signed
one-bit +1 encoding still fails and is retained separately from accepted native
bit-pattern execution.

The generic controlled SDK has an additional original failure. At width three,
`cirq.decompose` returns a list retaining controlled And-adjoint leaves.
Querying their original unitary recurses, as does native import of that same
unresolved original object. A list of source operations is therefore not proof
of a completely executable source decomposition. The already accepted native
specialized Negate control instead retains its literal controlled complement and
controlled-constant-load chain. Both control polarities have explicit source
failure witnesses. No provider source or production implementation was changed.

All source members now have a native owner or an explicit original/symbolic
failure disposition. Q-0211 and Q-0212 are accepted with these recorded limits.
The concentrated run passes 712 cases, with four existing paired NumPy overflow
warnings. The local integration run passes 17 cases. Both Cirq versions pass
20 focused cases. Strict typing, Black, Ruff and team scope pass. Production
architecture/API evidence from the preceding batch is reused because all eight
implementation owners are byte-identical. Five frozen source blobs and locked
Qualtran/Kickmix Simulator are verified. Dependency diagnostics remain 685 for
the repository and zero for this batch. Measured primitives and the original
models remain the separate tables already recorded above.

The Cirq 1.6.1 lane retains 1,512 upstream NumPy-2.5 array-shape deprecation
warnings from its recursive controlled-unitary failure witnesses. The Cirq 1.7
focused lane has no warnings. These warnings are retained in the receipt.
