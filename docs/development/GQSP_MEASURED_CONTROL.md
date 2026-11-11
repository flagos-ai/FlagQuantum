# GQSP through the original measured Z-power control path

This finite Compiler slice closes the actual control leaf path for Z powers:
`ZPowGate.get_ctrl_system` -> `CZPowGate` -> `And`, phase, `And.adjoint`.
Zero control uses the original single-bit `ControlledViaAnd` X sandwich.
`primitives/controlled_rotation.z_power_with_control_ir` composes the existing
control wrapper, measured CZ-power primitive and bare Pauli-power emitter.
No control engine, AND implementation, runtime, public export or IR schema is
added. Source is unchanged Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

## Which source path is compared

The reference MatrixGate used in earlier numerical tests is atomic and has no
leaf decomposition. A test retains its original DecomposeNotImplementedError;
that original absence cannot provide a physical gate-stream parity claim.

There is also a distinction within the original GQSP implementation. Direct
Cirq export of a ZPow uses Cirq's controlled gate. Calling the original
`GeneralizedQSP.decompose_bloq()` imports that Cirq graph through the original
converter and reaches Qualtran ControlledViaAnd/CZPow/And. This batch compares
**that unchanged Bloq decomposition**, including its measured erasure. The
direct Cirq unitary is an independent semantic oracle, not the measured trace.
We do not claim the two source representations have identical primitive counts.

The source walker uses original connections, control dispatch and adjoints.
Original And leaves are read through the existing reference Cirq decomposition
adapter. Atomic ZPow maps to one native phase. Signal SU2 uses the previously
verified common expansion, including the existing scalar-phase implementation
on a physical signal wire. This explicit atomic normalization is applied to
both sides. Counts below are not a comparison against raw zero-wire Cirq
GlobalPhase operations or synthesized Clifford+T gates.

## Binding and measured cleanup

The new helper takes target/exponent, optional control and activation, optional
clean work/result slot, and adjoint. A controlled call needs exactly one clean
work wire and a nonnegative result slot. A bare call uses neither; zero control
and non-boolean direction are validated explicitly. Adjoint negates the source
exponent before dispatch and retains measured erasure. It does not reverse
measurement instructions to manufacture an inverse.

The helper emits 18 instructions for positive control and 20 for zero control,
including four T/T-adjoint gates, one measurement and one reset. The control
wrapper itself uses no work; the single clean bit belongs to CZPow. The slot
may be reused after each feedback/erase finishes. Every forced measurement
history is compared with its unnormalized complex amplitude and branch weight;
work returns exactly to zero. Reset is deterministic given the preceding
projection. Classical storage contains the latest result after slot reuse.

The logical U is unitary after its internal feedback cleanup, so the existing
GQSP callback protocol is reused. Measured instructions execute through the
existing DynamicCircuit, keeping the same `circuit.append(op)` workflow. This
does not add a dynamic execution path or make static Circuit.state accept
measurements.

## Native executable append path

```python
import torch
from flagquantum.algorithms.primitives.generalized_qsp import GeneralizedQSP
from flagquantum.compiler.registers.primitives.controlled_rotation import z_power_with_control_ir
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic

def unitary(signal, ports, workspace, value, adjoint):
    (target,) = ports["q"]
    if value is None:
        return z_power_with_control_ir(target, 0.29, adjoint=adjoint)
    (work,) = workspace
    return z_power_with_control_ir(target, 0.29, control=signal, value=value,
                                   work=work, classical_bit=7, adjoint=adjoint)

operation = GeneralizedQSP.from_qsp_polynomial(
    (("q", 1),), unitary, (0.25, 0.1, 0.2), negative_power=1,
)
circuit = DynamicCircuit(4, dtype=torch.complex128)
circuit.h(1)
circuit.append(operation.make_on(signal=1, target=(3,), workspace=(0,)))
result = run_dynamic(circuit, shots=2, seed=23)
for state in result.final_states:
    logical = state.reshape(2, 2, 2, 2).permute(1, 3, 0, 2).reshape(4, 4)
    assert torch.count_nonzero(logical[:, 1:]) == 0  # work and idle
    assert torch.allclose(state.abs().square().sum(), torch.tensor(1.0, dtype=torch.float64))
```

## Same-normalization measured counts and separate QEC models

For the example P, the source-normalized and native instruction tuples match
exactly, including gate order, operands, angle parameters and measurement
feedback metadata. There is one clean work bit, one reusable classical slot,
and one idle probe bit. The idle probe is not workspace.

| Laurent offset k | Native instructions | Source-normalized instructions | Measured T/T† | Measurements/resets | Source QEC And / rotation / Clifford / measurement |
|---|---:|---:|---:|---:|---|
| 0 | 61 | 61 | 8 | 2 / 2 | 2 / 11 / 6 / 2 |
| 1 | 59 | 59 | 8 | 2 / 2 | 2 / 11 / 4 / 2 |
| 6 | 61 | 61 | 8 | 2 / 2 | 2 / 15 / 2 / 2 |

The shared QEC query prices actual Z-power child identities, then the existing
GQSP component model aggregates them. All seven categories match the original
QECGatesCost independently of measured primitive totals. Source models retain
And as a logical category; measured T/T† counts expand it. No T-equivalent
synthesis price is inferred from a continuous rotation budget. Model peak width
is three logical bits; the fourth physical bit in tests is only an idle probe.

Nine full GQSP scenarios use n=2/3/4 coefficients and k=0/1/6. Four basis states
plus a complex coherent state are checked per scenario over all 2^(n-1) And
measurement histories: 210 unnormalized branches and 45 actual dynamic-runtime
trajectories. Source/native full amplitudes and clean work/idle zeros agree.
Additional tests check both control polarities and adjoints against original
ZPow controls, original MatrixGate absence and invalid storage/operand bindings.
Arbitrary U leaves, symbolic inputs, high-degree conditioning and physical
synthesis remain outside this acceptance slice; full parent files remain open.

## Local verification record

Implementation `34bbf789` passed 21 new tests, 229 focused tests (7.72 s),
522 related tests (5.25 s) and 64 integration tests (4.05 s). The touched
controlled_rotation module's 28 statements and ten branches were covered at
100%. Strict Python 3.12 typing for two files, Black, Ruff, Compiler scope,
architecture, IR and public API checks passed. The focused/related runs retained
48 previously independently reproduced matmul warnings from older multi-register
fixtures; the new tests had no warnings. Fourteen original blobs matched the
locked CSV, and 27 unchanged original tests passed (1.14 s), retaining six
marker warnings and six original autoray matmul warnings. The native dynamic
Markdown example was executed.

The required default run recorded 21507 passed, 300 skipped, 1037 warnings and
seven failures (152.30 s). All seven failed IDs matched the established
independent baseline; no new failures were observed. This maxfail-stopped run
is not green or a complete-suite result. Per-primitive histograms, normalized
source equality and separate source QEC categories are in
`ftqc-reference-env/g2-gqsp-measured-control-counts.json`; other evidence shares
the `g2-gqsp-measured-control-` prefix. Publication is verified separately in
the external progress ledger.

The earlier direct native cphase callback remains a different explicit child
implementation. Generic GQSP still receives its U emitter from the caller;
this batch does not silently replace every callback with a measured one.
Only the Z-power source-dispatch slice above is accepted here.
