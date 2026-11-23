# Approximate Fourier constructor closure

## Source and native ownership

Complete qft/approximate_qft.py and its original tests remain frozen at
Qualtran v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389. The intake verifies
six exact Git blobs and actual locked Kickmix 0.1 Simulator availability.

| Source definition | Native owner |
|---|---|
| ApproximateQFT fields and default | Compiler formats/legacy_approximate_qft_factories.py reuses shared ceil/log2 metadata and the original Boolean reversal default. |
| Signature | Shared unsigned MSB-first q and unsigned fractional phase_grad retain caller ownership. Zero-width gradient ports are omitted exactly as on the source wire. |
| decompose_from_registers | phase/approximate_qft_component.py retains each H, reversed active prefix, controlled shift-one phase-add window and final reversal through existing helper definitions. |
| AddIntoPhaseGrad dependency | Existing scaled_phase_component.py and controlled arithmetic own measured work, feedback and cleanup. No alternate numerical kernel is introduced. |
| from_epsilon and build_call_graph | Existing Algorithms approximate_qft_resource_model.py retains concrete and symbolic precision bounds, source calls and independent resource formulas. |
| Both examples and DocSpec | The original six-bit and epsilon-derived fifty-bit examples resolve to the same native Program as explicit metadata. |

The complete native component is reused through existing binding and ordinary
circuit.append(op). Concrete source calls, operands and bound events are
compared across both reversals and stored library depths zero, one and two.
Complete joint complex operators are compared for bounded data/gradient sizes.
Exact native Program and bound instruction equality permit reuse of accepted
Fourier tests for all applicable small measurement histories, coherent spectator,
noncontiguous binding, inverse recovery and zero private work.

## Accounting and original boundaries

Primitive inventories count the bound native instruction stream independently
of the original analytical Toffoli/T-equivalent and peak-qubit models. One-bit
gradient calls have empty x input and emit no primitive or workspace, while the
original model retains negative Toffoli estimates. These estimates are never
interpreted as physical instructions.

The original zero-data decomposition raises KeyError for q. Multi-bit data
with zero precision raises KeyError for phase_grad. Direct symbolic widths,
including Sympy Integer, refuse decomposition. Bodyless native declarations
retain those direct boundaries and physical binding rejects them. Original
wire normalization converts concrete Sympy Integer metadata to Python int.
Single-bit data with zero gradient remains executable with one H.
Fully symbolic physical execution and signed storage are separate boundaries.

## Acceptance receipts

The concentrated constructor/Fourier regression passes 371 cases. Affected
local integration passes 145 cases. Thirteen unchanged original precision and
complexity tests pass, with nineteen unselected original cases. Earlier full
Fourier and original-source receipts are reused for unchanged execution owners.
Strict checking covers both changed Python files. Black, Ruff, Compiler scope,
language, architecture, multilevel IR and public API checks pass. Dependency
checking retains exactly the existing 720 findings. Receipts use prefix
g2-approximate-constructor and date 20261123.

At data width three and precision two, actual execution emits 16 T/T-dagger,
28 CX and four measurements/resets on seven physical wires. The original
aggregate estimates zero Toffoli and five qubits. At width six and precision
five, execution emits 112 T/T-dagger, 232 CX and 28 measurements/resets on
nineteen wires. The separate source model estimates eighteen Toffoli and
eleven qubits. Per-call source operand equality is verified independently
of those aggregate cost estimates. At width three and precision one, actual
execution has three H and one SWAP on four wires, while the source model
retains negative four Toffoli and five qubits.
