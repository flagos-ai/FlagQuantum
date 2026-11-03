# SelectedMajoranaFermion

Status: proposed. API-owner review pending.

## User workflow

Replace the two dense Majorana operators in QC-003 with named, executable
algorithm components. Preserve the compact SELECT as an independent reference.
Both expanded operators must remain identifiable in the IR and drawer.

Proposed import:

```python
from flagquantum.algorithms.primitives.multiplexers import SelectedMajoranaFermion

operation = SelectedMajoranaFermion.make_on(
    selection=(p, spin),
    target=system,
    control=(control,),
    target_gate="Y",
)
```

Reuse the Qualtran class, factory and operand names. FlagQuantum qubit indices
and Pauli names replace Cirq objects. The operation returned by the factory
needs a Circuit binding path. Review the existing construction APIs before
adding any public Circuit method. The explicit constructor retains
`selection_regs`, `control_regs`, and `target_gate`; its register descriptor and
serialization are part of the contract review.

## First implementation

Append the bound component with `operation.append_to(circuit)`. The component
records a logical instruction with register widths and the Pauli. The
[deferred IR contract](FQ-DEFERRED-MAJORANA-IR.md) defines how the compiler
materializes it. The factory accepts a flat binary address, which covers
QC-003's two one-bit selection registers.

## Circuit integration

Evaluate `Circuit.compose(other, qubits=...)` as the first integration path
for reusable Majorana programs. It already binds operands, appends IR
instructions, and returns the receiving circuit. A general
`Circuit.append(operation)` entry point requires a concrete need beyond this
component and a separate API review.

Build the logical operation with a canonical operand order of control,
selection, then target. Store role widths and Pauli parameters independently
of absolute qubit IDs. This lets composition remap instruction operands
without leaving stale qubit IDs in the parameters.

Construction validates the binding and records the instruction. Serialization
preserves its logical representation. Compilation or execution materializes
the dense reference matrix, with a limit of ten operand qubits.

Acceptance includes construction and JSON round-trip without matrix
allocation, composition onto reordered qubits, and complex-amplitude parity
through the normal execution entry points against Qualtran 0.7.0.

## Semantics

Selection registers encode unsigned integers in most-significant-bit order.
Combine their values in row-major order using the declared iteration lengths.
For selected mode j, apply Z to modes below j and the target Pauli to mode j.
An absent control enables the operation unconditionally. Bound control qubits
enable it when all are one. Preserve relative phase on superposed selections and
controls. Define unused binary addresses as identity in the FlagQuantum
extension and check upstream behavior separately before claiming parity there.

Validate register names, widths, iteration lengths, target size and disjoint
qubit bindings before modifying a circuit. Start with X, Y and Z target gates.
Record the concrete selection dimensions, controls, targets and Pauli in the
structured operation so execution does not depend on reconstructing Python objects.

## Package layout

```text
flagquantum/algorithms/
  primitives/
    multiplexers/
      __init__.py
      selected_majorana_fermion.py
      select_pauli_lcu.py           # Follow-up
    data_loading/                  # QROM and QROAM follow-ups
    state_preparation.py           # Existing module
  chemistry.py                     # Existing module
```

Create only `primitives/multiplexers/` in the first implementation. Keep one
public construct per implementation module and export it from the domain
package. Add tests in a matching `tests/unit/algorithms/primitives/multiplexers/`
directory. Keep the executable paper example under
`examples/plot_gallery/quantum_circuits/`.

When a domain outgrows its existing module, migrate it to a package with the
same import path and re-export its existing public names. Keep paper-specific
assembly in examples and shared decomposition in the compiler.

## Ownership and sequencing

- `algorithms/primitives/multiplexers` owns the user-facing algorithm component.
- Existing core IR owns the persistent instruction contract. The compiler owns
  decomposition and ancilla handling. Keep the logical operation available for
  drawing and resource analysis.
- Reuse existing reversible-control primitives after checking their clean-ancilla
  preconditions. Document the first decomposition's cost separately from
  Qualtran's unary-iteration implementation.
- Keep QC-003 migration and the generic `SelectPauliLCU` in separate changes.
  Existing public imports and serialized programs continue to work.

## Acceptance evidence

Compare complex amplitudes against an independent reference for X, Y and Z,
including control and selection superpositions, multi-register addressing,
unused addresses and disabled control. Check malformed bindings fail before
append. Check serialization round-trip, decomposition equivalence and clean
ancilla restoration. Then migrate QC-003 and compare all 512 logical basis
inputs against its independent compact SELECT. Verify the two module hotspots
resolve to separate source calls.

Reference: [Qualtran SelectedMajoranaFermion](https://github.com/quantumlib/Qualtran/blob/main/qualtran/bloqs/multiplexers/selected_majorana_fermion.py).
Freeze the upstream reference revision when opening the implementation PR.

## Python, Rust, and C++ responsibilities

| Language | Owner and scope |
| --- | --- |
| Python | Owns the public API, register binding and validation, reference decomposition, compiler and runtime orchestration, and drawer. Qualtran-compatible component names live in the algorithm library. |
| Rust | Optional native acceleration for measured FTQC analysis bottlenecks. Begin with hierarchical resource counting if profiling supports it. Consider IR traversal or lowering separately with their own benchmarks. |
| C++ | Owns optimized numerical kernels behind the existing simulation dispatch. Reuse the current native CPU extension and Torch tensor interfaces for supported operations. Add a kernel only where measured execution costs justify it. |

### Shared IR contract

Python constructs the canonical, versioned FlagQuantum IR. A structured
operation records its opcode, parameters, register roles, ordered qubit
bindings and controls. Logical operation identity and source mapping survive
lowering so the drawer can show the compact and expanded circuits.

Rust receives batches of structured operations through a private adapter and
returns counts or analysis results with backend provenance. The adapter projects
from the canonical IR; the projected representation is internal and versioned
with its consumer. The compiler lowers the same logical program into supported
execution operations. The runtime dispatches these as kernel operands and
tensor buffers to C++. C++ operates on this execution representation without
deserializing the high-level program. Schema or opcode changes belong to the
existing IR review process.

Verify serialization round-trip, Python/Rust analysis parity, and Python/C++
execution parity on the same fixtures. Check wire order, phases, auxiliary
qubit restoration, unsupported cases, and deterministic results. Explicit
backend requests report missing extensions or unsupported features. Automatic
selection records the backend that ran.

### Package layout

```text
flagquantum/
  algorithms/primitives/multiplexers/  # User-facing algorithm modules
  core/ir/                            # Canonical logical IR
  compiler/                           # Lowering and synthesis
  drawer/                             # Compact and expanded views
  ftqc/                               # Proposed resource analysis and private Rust adapter
  simulation/native_cpu/              # Existing C++ extension and integration
native/
  ftqc/                               # Proposed optional Rust crate
tests/
  unit/algorithms/primitives/multiplexers/
  integration/                        # Shared IR and cross-backend conformance
```

Create each new directory when its first implementation lands. Keep the
existing C++ build and dispatch ownership. Any future C++ extension change
reuses that boundary.

### PR sequence

1. **Majorana component and IR contract.** Implement the reference Python
   path and test register binding, serialization, and exact amplitudes.
2. **QC-003 migration.** Replace the expanded side's hand-written matrices,
   compare all 512 basis inputs, and verify both drawer hotspots.
3. **Pauli LCU and resource baseline.** Add `SelectPauliLCU`, then PREPARE
   and block encoding in separate changes. Establish the Python hierarchical
   count on a small circuit and profile a larger chemistry workload.
4. **Rust analysis pilot.** Port one demonstrated hotspot. Compare counts
   and errors against Python, and report end-to-end wall time and peak memory,
   including conversion overhead.
5. **C++ execution optimization, if profiling warrants it.** Reuse the
   current kernels first. Any new kernel requires amplitude parity and an
   end-to-end benchmark through the existing runtime. This can run independently
   of the Rust pilot once the reference fixtures exist.
6. **Optional native distribution.** Add build and wheel coverage after
   correctness and performance gates pass, including installs without the
   optional Rust extension.

This is a proposed FlagQuantum design. Qualtran is the reference for component
APIs and algorithm semantics. Its publicly documented integrations do not
establish an equivalent three-language backend contract.
