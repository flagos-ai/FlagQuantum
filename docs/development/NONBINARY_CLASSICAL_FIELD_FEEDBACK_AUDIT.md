# Nonbinary classical field feedback acceptance

## Scope and ownership

Proposal 111 was explicitly approved on 2026-11-20. This closes the concrete
nonbinary classical field dependency of complete `controlled.py`,
`classically_controlled.py` and `gate_with_registers.py`. The accepted contract is
`contracts/nonbinary-classical-field-contract.toml`. The execution classification
is `single_device_fast_path`.

Core reuses one structural field metadata validator for scalar fields and packed
polynomials, with independent classical and quantum admission. Numerical field
construction remains in Simulation. Compiler reuses the existing read opcode,
linear successor, snapshot conditions, stored calls, slot binding and component
builder. Its only additional value validation checks every packed coefficient
against the field order, including control activations. No extra public entry
or opcode is introduced. The example continues to use `circuit.append(op)`.

The field word is unsigned. Polynomial coefficients appear in descending degree
order and each coefficient uses `(order - 1).bit_length()` MSB-first bits. Arrays
use row-major order. A GF(3) degree-one polynomial `2*x+2` therefore carries
stored word 10. Core validates canonical integer metadata and widths. Simulation
retains numerical primality and irreducibility construction.

## Frozen comparison and retained source defect

The reference is unchanged Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`. Receipts verify the ledger blobs for
`controlled.py`, `data_types.py` and `classically_controlled.py`, the Qualtran
version and native Kickmix 0.1 `Simulator` availability.

Six concrete fixtures cover GF(3), GF(5), GF(9), degree-one polynomials over
GF(3)/GF(5), and degree-zero polynomials over GF(9). All 902 legal activation/input
pairs compare bit encoding and original classical callback actions. Scalar and
constant-polynomial tensor blocks also match the full complex spectator-state
execution. The existing child graph is compared primitive by primitive,
including operands. All legal activations/inputs preserve a clean work bit.

The original nonbinary polynomial tensor uses the polynomial's radix-field-order
integer as its active index. Its coefficient bit encoding uses a different
index. For GF(3), `2*x+2` has integer 8 and packed word 10. For GF(5), `4*x+4`
has integer 24 and packed word 36. Tests preserve all four discrepant active and
inactive tensor blocks. Native typed execution follows original numerical
callbacks and bit codecs. Whole-wrapper tensor equivalence is restricted to
cases where those original representations agree.

The original classical wrapper has no physical decomposition. Cost evaluation
raises either its absent-decomposition error or an earlier unhashable field
activation error. These failures are asserted separately from child primitive
and model comparisons. Original sources are unmodified.

## Execution and compatibility

The six fixtures cover scalar, shaped and empty nested transport, nominal output
types and repeated-call slot reuse. Reads preserve the same-type linear
successor and MSB snapshots. Every unused coefficient encoding is rejected in
entry values and activations, together with negative, boolean and out-of-width
stored words. A mixed two-element field array and polynomial group exercises
all 81 legal combinations with row-major feedback.

Each fixture composes the stored field predicate with a measured child CBit.
Both extreme legal activation words, every legal input and both X-measurement
outcomes are checked against unchanged source callbacks and full complex
reference branches, totaling 240 histories. Representative bindings execute
through `DynamicCircuit.append`. The example
`python -m examples.algorithms.typed_classical_feedback` exercises signed words,
binary fields/polynomials, GF(3), and packed GF(3) polynomials through that entry.

Pre-extension fixtures compare exact Program representations, Circuit JSON,
content hashes, aliases and classical slots for binary fields, polynomials and
accepted integer controls. Existing Proposal 108/109/110 fixtures also pass.

## Resources

| Scenario | Actual emitted instructions | Clean work | Classical slots | Original source estimate |
| --- | --- | --- | --- | --- |
| Repeated quantum child under a field word of width w | 4 CX, 2 phase, 2 H, w classical initializers | 1 reused bit | w | Child 2 T and 6 Clifford, wrapper model absent |
| Field predicate and measured child | 1 H, 1 measurement, 1 guarded X, w initializers | 0 | w + 1 | Wrapper model absent |
| Runnable scalar/polynomial example | 1 guarded X, w initializers | 0 | w | Wrapper model absent |

Reads add zero physical instructions and zero slots. Resource observations and
original estimates are recorded separately in `g2-nonbinary-field-evidence.json`.

## Validation and remaining dependencies

The concentrated field/control regression passes 4454 cases with zero skips
and 30 existing paired source/native overflow warnings. All 17 local integration
cases pass. Strict typing covers seven changed Python files. Black, Ruff,
classified ownership scopes, architecture, IR and public API checks pass.
The example executes successfully. Three frozen source blobs and both reference
versions are verified. Dependency inspection reports 687 existing repository
findings and zero on batch paths.

Nonbinary quantum execution, unresolved symbolic physical descriptors,
conditional measurement/reset/writes and generic foreign object/serialization
contracts retain their separately tracked boundaries. Source defects remain
explicit. This dependency acceptance alone does not accept the complete
controlled or register-gate parent files.

The default smoke/unit tier stops after 434 passes and 51 skips at the existing
`test_every_module_with_examples_is_covered` failure. Its nine reported modules
and failing test are byte-identical to pre-batch head
`717f2b39dec59af49f3d1d1dfecad1e78df67a1e`, matching the previously recorded
baseline. `g2-nonbinary-field-default-baseline.json` preserves that comparison.
Whole-repository passing status is not claimed.
