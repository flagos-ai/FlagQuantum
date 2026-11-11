# Generalized QSP named-register operation

`algorithms/primitives/generalized_qsp.py` binds the finite explicit-P/Q slice of
Qualtran v0.7.0 `GeneralizedQSP`, pinned to
`bcb17bb25360ed68ce3bc084645db3e929799389`. It composes the existing Simulation
phase recursion and Compiler signal emitter; it does not duplicate either.
This is an internal `single_device_fast_path` operation with no new export,
IR schema, execution engine or dependency.

## Ownership and ports

The operation snapshots P and Q in ascending degree order, retaining the source
minimum-degree check and lazy equal-length failure. As in the source, phase
computation is cached. A complementary pair is the caller's responsibility.
No automatic polynomial repair or complementarity certificate is introduced.

The declared unitary layout maps a flat target tuple to named, big-endian ports
in declaration order. Zero-width ports retain their empty tuples. Signal is a
separate wire; the name `signal` is reserved. Clean workspace is explicit and
disjoint from signal and target. It belongs to U's emitter, which must restore
it after **every** invocation. Both forward/inverse and control semantics are
callback obligations; the operation does not prove arbitrary callback behavior.

The callback receives `(signal, ports, workspace, control_value, adjoint)`;
None means uncontrolled. Each call gets a fresh port mapping. Compiler checks
that emitted instructions remain within the physical binding. The append
adapter validates the circuit width, preserves its batch size and composes the
same emitted IR. A failing emitter does not partially append its gate prefix.

This slice handles explicit finite coefficient pairs, concrete layouts and the
Compiler's nonnegative Python-integer Laurent offset. The finite polynomial factory, precision declarations and resource model are
now covered by [GENERALIZED_QSP_FACTORY_MODEL.md](GENERALIZED_QSP_FACTORY_MODEL.md).
Symbolic metadata, shaped/dtype signatures and physical synthesis remain open. Continuous rotations have no
Clifford+T synthesis-error claim. Numerical input coefficients are classical
values, not trainable tensors. There is no gradient or scalability claim.

## Native executable path

No reference package or example adapter is needed:

```python
import torch
import flagquantum as fq
from flagquantum.algorithms.primitives.generalized_qsp import GeneralizedQSP
from flagquantum.compiler.registers.primitives.pauli_rotation import pauli_power_ir
from flagquantum.core.ir import Instruction

def unitary(signal, ports, workspace, value, adjoint):
    (target,) = ports["system"]
    gates = pauli_power_ir("z", target, -0.29 if adjoint else 0.29,
                           control=None if value is None else signal)
    if value == 0:
        flip = Instruction("x", (signal,))
        return (flip, *gates, flip)
    return gates

operation = GeneralizedQSP((("system", 1),), unitary,
                          (0.5, 0.5), (0.5, -0.5), negative_power=1)
circuit = fq.Circuit(3, dtype=torch.complex128)
circuit.append(operation.make_on(signal=1, target=(0,)))
assert torch.count_nonzero(circuit.state().reshape(4, 2)[:, 1]) == 0
assert torch.allclose(circuit.state().abs().square().sum(),
                      torch.tensor(1.0, dtype=torch.float64))
```

## Verification scope

`tests/unit/test_generalized_qsp_operation.py` uses two named system registers
(widths two and one), physically scrambled as target `(4,1,3)`, signal 2,
workspace 0 and idle bit 5. U combines non-Hermitian X/Z powers, including X's
nonzero global shift. The emitter actually computes target bits into clean
workspace, phases them and uncomputes. Every one of 16 basis inputs and a
complex coherent input is executed for each of 12 length/real-complex/offset
scenarios. Full amplitudes match the unchanged source GQSP circuit, the P/Q
blocks match an independent Laurent evaluator, and work plus idle are exactly
zero. There are no measurements; each case has one coherent execution branch.

Binding emits exactly the existing Compiler stream with unchanged source
phases: **zero additional physical instructions**. Callback counts separately
follow n rotations / max(0,d-k) zero-controlled U / min(d,k) controlled U† /
max(0,k-d) U†, d=n-1. These component counts are not primitive counts. The source
first-level control and generic-leaf boundary documented in
[GENERALIZED_QSP_EMISSION.md](GENERALIZED_QSP_EMISSION.md) remains unchanged;
this wrapper does not establish arbitrary-U primitive parity.

Other scenarios cover coefficient snapshots, source constructor/lazy failures,
invalid bindings, callback wire escape, atomic failed append, fresh callback
mappings, cached phases and existing batched append metadata. The source
constant/tail-zero failures and high-degree numerical boundaries remain open.

## Measured counts and local evidence

For P=(.25,.1,.2) and the three-system-bit, one-work-bit U in the focused test,
the native stream was counted directly. The probe idle bit is not workspace.
Binding contributes zero gates beyond the existing emitter.

| Offset k | Measured instructions | Clean work | Measurements | Source model SU2 / C[0]U / CU† / U† |
|---|---:|---:|---:|---|
| 0 | 45 | 1 | 0 | 3 / 2 / 0 / 0 |
| 1 | 43 | 1 | 0 | 3 / 1 / 1 / 0 |
| 6 | 93 | 1 | 0 | 3 / 0 / 2 / 4 |

These totals include physical scalar-phase implementation instructions and
workspace compute/erase CNOTs. They are not synthesized Clifford+T estimates.
The complete per-primitive histogram is in
`ftqc-reference-env/g2-gqsp-operation-counts.json`.

Implementation `1d6c2584` passed 24 new tests, 124 focused tests (5.26 s),
417 related tests (5.19 s), and 64 integration tests (5.13 s). All 48 new
statements and six branches were covered. Strict Python 3.12 typing, Black,
Ruff, Algorithms scope, architecture, IR and public API checks passed. Six
original blobs matched the locked inventory; 17 unchanged original call-graph
tests passed (3.01 s) with six original unknown-marker warnings. The native
Markdown example was executed.

The focused and related runs each retained 48 matmul runtime warnings from the
original MatrixGate check and existing dense polynomial evaluator. An isolated
probe importing neither the new operation nor its emitter reproduced the
warnings on the same finite unitary. Scalar arithmetic independently confirmed
unitarity, and the matmul result was also finite and unitary; full source/native
amplitudes are checked finite before comparison. The probe evidence is
`g2-gqsp-operation-matmul-probe.json`. Warnings were not suppressed and this
observation does not diagnose the underlying numerical library.

Default-suite baseline comparison and verified publication status are recorded
in the external progress ledger and `g2-gqsp-operation-*` evidence. Baseline
failures are not a green default run. Full parent-file acceptance remains open.
