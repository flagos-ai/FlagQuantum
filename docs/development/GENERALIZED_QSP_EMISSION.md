# Finite generalized QSP emission

`compiler/registers/phase/generalized_qsp.py` owns the finite signal sequence
from Qualtran v0.7.0 `GeneralizedQSP.decompose_from_registers`, pinned to
`bcb17bb25360ed68ce3bc084645db3e929799389`. This is a
`single_device_fast_path` slice. Simulation owns complement and phase computation;
Compiler receives concrete phases and emits existing IR. No exports, schemas,
new execution engine or dependencies are introduced.

## Binding and ownership

`generalized_qsp_ir(signal, operands, theta, phi, lambd, unitary=...,
negative_power=k)` emits the first SU2 rotation, then interleaves each later
rotation with one unitary call. The first min(k, degree) calls are positively
controlled U adjoints; the rest are zero-controlled U. Remaining inverse calls
are unconditional. Lambda belongs only to the first rotation. Absolute phase
is retained by the existing SU2 emitter, including its scalar-phase gates.

Signal is one bit disjoint from the ordered operand tuple. The bound callback
receives `(control_value, adjoint)`, where None means no control. It owns U's
register order, specializes control and adjoint, and restores any workspace
included in operands. The emitter rejects wires outside the binding. It does
not prove an arbitrary callback's unitarity, workspace restoration, or correct
control implementation. The caller must supply a coherent unitary.

Finite nonnegative Python integer k and equal phase lengths of at least two
are the supported slice. Invalid lengths, nonfinite angles, duplicate wires,
negative k, booleans and fractional k are rejected. These guards are native
boundary validation, not a claim that the original constructor rejects the
same inputs. Symbolic phases, original polynomial factories, arbitrary typed
register signatures, synthesis precision and whole-wrapper acceptance remain
open. SU2's precision metadata is checked on the reference; this emitter emits
continuous rotations and makes no synthesis error or Clifford+T count claim.

## Executable append path

This reuses the existing internal append example adapter. No reference package
is needed to run it.

```python
import torch
import flagquantum as fq
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.phase.generalized_qsp import generalized_qsp_ir
from flagquantum.compiler.registers.primitives.pauli_rotation import pauli_power_ir
from flagquantum.core.ir import Instruction
from flagquantum.simulation.numerics.qsp_complement import qsp_complementary_polynomial
from flagquantum.simulation.numerics.qsp_phases import qsp_phase_factors

p = (0.25, 0.1, 0.2)
q = qsp_complementary_polynomial(p)
theta, phi, lambd = qsp_phase_factors(p, tuple(q))

def unitary(value, adjoint):
    gates = pauli_power_ir("z", 1, -0.29 if adjoint else 0.29,
                           control=None if value is None else 0)
    if value == 0:
        flip = Instruction("x", (0,))
        return (flip, *gates, flip)
    return gates

gates = generalized_qsp_ir(0, (1, 2), tuple(map(float, theta)),
                          tuple(map(float, phi)), lambd,
                          unitary=unitary, negative_power=1)
circuit = fq.Circuit(3, dtype=torch.complex128)
circuit.append(FieldOperation(3, gates))
assert len(gates) == 25
assert torch.count_nonzero(circuit.state().reshape(4, 2)[:, 1]) == 0
```

## Parity and resource boundaries

The tests obtain the original first-level Cirq decomposition and compare every
SU2 angle, operand, control value and adjoint call in order. Both source
`Controlled` and zero-control `ControlledViaAnd` records are retained by the
comparison. Replacing each source stage with the existing native SU2/Pauli-power
emitters gives exactly the same Instruction stream as the new orchestrator.
This proves the orchestration at those shared component boundaries. It does
**not** assert that specializing a reference MatrixGate to a native Pauli power
reproduces every primitive in upstream's generic controlled-MatrixGate lowering.
That generic leaf decomposition remains outside this acceptance slice.

Eighteen cases use lengths 2/3/5, k=0/1/6, and non-Hermitian X/Z powers. X has a
nonzero scalar shift, making controlled phase and adjoint errors observable.
All four basis inputs and a complex coherent input run through circuit.append;
full complex amplitudes match original GQSP, both polynomial blocks match the
independent Laurent evaluator, and the idle third bit remains exactly zero.
There are no measurements, so there is one coherent execution branch. A callback
replacement scenario keeps the consumer unchanged.

For the actual example P=(.25,.1,.2), these measured Instruction counts include
all scalar-phase implementation gates. Work/measurement counts are zero;
the third bit is an idle cleanliness probe, not allocated work. X uses exponent
.29 and global_shift=.17; Z uses exponent .29 and zero shift.

| k | Measured Z instructions | Measured X instructions | Source model SU2 / zero-controlled U / controlled U† / U† |
|---|---:|---:|---|
| 0 | 27 | 33 | 3 / 2 / 0 / 0 |
| 1 | 25 | 31 | 3 / 1 / 1 / 0 |
| 6 | 27 | 57 | 3 / 0 / 2 / 4 |

The source model is n / max(0,d-k) / min(d,k) / max(0,k-d), d=n-1.
Tests separately compare the original build_call_graph total and actual native
callback counts. A component model count is not a physical primitive count.
Instruction equality above and full amplitude checks are separate evidence.

Run `tests/unit/test_generalized_qsp_emission.py` with the locked reference
available for parity, or execute the example for the native-only path. Original
complement failures and high-degree conditioning boundaries remain as recorded
in [QSP_COMPLEMENT_PHASES.md](QSP_COMPLEMENT_PHASES.md). No reference code was
modified or silently skipped.

## Verification evidence

The implementation commit is `2291b5cd`. The batch passed 27 new tests,
100 focused tests (4.80 s), 393 related tests (4.19 s), and 64 integration tests
(4.61 s). All 29 new statements and 12 branches were covered. Strict Python
3.12 typing, Black, Ruff, Compiler scope, architecture, IR and public API checks
passed. Six source blobs matched the locked inventory; 17 unchanged source
call-graph tests passed (0.60 s) with six original unknown-marker warnings.
The actual Markdown example was executed successfully. Default-suite comparison
and publication status are recorded in the external progress ledger and
`ftqc-reference-env/g2-gqsp-emission-*` evidence, without treating known baseline
failures as a green suite.
