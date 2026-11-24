# Stored QROM constructor audit

## Scope and ownership

This batch closes the concrete serialized constructor path of the complete
`data_loading/qrom.py` source with direct `qrom_base.py`, table and unary-iteration
dependencies. The reference remains Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. No reference file is modified.
Classification is `single_device_fast_path`.

The format adapter validates serialized dataset and dimension metadata and then
calls the existing `QromTable` and `qrom_program` owners. It preserves explicit
selection widths, target widths, target array shapes and grouped control count.
It does not allocate simulation state or select an execution backend.

All control, selection and target words are caller-owned THRU ports. Bits are
MSB first and array elements are row-major. Targets may start dirty. Lookup XORs
the selected row into them. Predicate and carry components own their private
clean work, exact measured feedback and erasure. Their work returns to zero.

## Complete concrete dependency disposition

All three concrete original examples retain their complete metadata at serialized
depths zero and one. Multidimensional selections, multiple datasets, target
arrays, constant-data folding and zero through three controls reuse the existing
bounded unary-iteration and XOR-word owners. Whole recursive source calls and
operands are compared before primitive lowering.

Physical replacement tests bind the actual reconstructed entry and require
identical primitives, measurement feedback slots, allocation events, releases
and logical outputs to the previously accepted QROM Program. Full small dirty-
target complex operators and all-history erasure certificates are checked. Prior
source component tests cover source primitive calls and deeper dependency
ordering. Original resource formulas and primitive inventories remain separate.

## Explicit boundaries

The original symbolic example constructs a valid shape-only QROM, but its
serializer raises `Object arrays cannot be saved when allow_pickle=False`.
Native `QromTable.from_shape` and bodyless `qrom_program` remain the existing
shape-only owners. This adapter does not add pickle support or invent omitted
data. That serialized symbolic constructor boundary remains partial.

The original programmable rotation array similarly serializes no angle, kappa
or gate constructor arguments, and its original reconstruction fails for missing
required keyword arguments. Its stored decomposition does retain QROM children.
The QROM adapter supplies that direct dependency. Restoring that stored body and
its constructor metadata remain separate work.

Existing mixed-singleton nonuniform QROM indexing refusals and unresolved physical
data remain explicit. Irregular dataset or symbolic object-array serialization
failures are retained at the original boundary. Unsupported physical dimensions
continue to fail through the existing table/layout owners. Ordinary execution
uses the existing `circuit.append(op)` path.

The four previously registered source/native metadata admission differences are
still open. They are not repaired by successful concrete reconstruction.

| Source metadata | Remaining native difference |
| --- | --- |
| Datasets with different row counts | Source defers the common-shape failure; native table construction rejects immediately |
| Declared target shape `(3,)` on an array of shape `(3, 2)` | Source retains the declaration; native table construction rejects the inconsistent suffix |
| Scalar first dataset followed by a vector | Source empty-prefix sentinel can select the vector shape with a scalar-derived selector width; native table rejects the mismatch |
| Negative control width | Source can retain an opaque negative-width declaration; native metadata rejects it before binding |

Q-0480, Q-0481 and the generic QROM constructor identifier remain partial. No
whole-file admission parity is claimed.

## Acceptance receipt, 2026-11-24

The concentrated regression passes 181 cases with no skips. Affected local
integration passes 145 cases. All unchanged original file tests were exercised.
The 45 non-notebook checks pass, with three source-owned skips and three expected
failures. The original notebook passes separately after propagating the existing
dependency paths into its kernel environment. Initial notebook attempts recorded
the sandbox socket refusal and missing kernel module. No upstream code or test
was edited.

Four frozen Git blob identities match the locked upstream tree. Qualtran 0.7.0
and the actual Kickmix 0.1 Simulator are verified. Strict mypy checks both new
Python files. Black, Ruff, Compiler scope, language, architecture, multi-level IR
and API checks pass. Dependency analysis retains the same 720 existing findings.

| Original example | Actual primitives and used wires | Independent original model |
| --- | --- | --- |
| Five rows, one dataset | 12 T/Tdag, 26 CX, three measurements, eight wires | Three And, 13 Cliffords, three measurements, eight logical wires |
| Five rows, two datasets | 12 T/Tdag, 33 CX, three measurements, 12 wires | Three And, 20 Cliffords, three measurements, 12 logical wires |
| Three-by-three rows, two datasets | 28 T/Tdag, 77 CX, seven measurements, 23 wires | Seven And, 44 Cliffords, seven measurements, 23 logical wires |

Conditional CZ, reset and the remaining Clifford primitives are retained in the
machine-readable inventory. The shared component hierarchy keeps source XOR
calls separate from their primitive expansion. No constructor coverage increment
is claimed for the still-partial generic QROM identifier.
