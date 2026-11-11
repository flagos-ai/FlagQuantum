# Generalized QSP finite factory and declared resource model

This slice follows Qualtran v0.7.0 `GeneralizedQSP.from_qsp_polynomial`,
`signal_rotations`, `_eps_per_rotation` and `build_call_graph`, pinned to
`bcb17bb25360ed68ce3bc084645db3e929799389`. Algorithms owns composition,
Simulation owns grid/complement/phase numerics, and Compiler owns physical
emission and shared cost aggregation. No source patch, new runtime, dependency,
public export or protected schema is introduced.

## Factory and failure ordering

`GeneralizedQSP.from_qsp_polynomial(layout, emitter, p, ...)` uses the existing
finite-grid assertion when verify=True, then the existing root-selected
complement with the same verify/verify_precision arguments, then construction.
The result snapshots P and Q and computes phases lazily as before. Original
constant empty-root interpolation, trailing-zero verification and over-bound
grid failures remain visible; no trimming, perturbing, sorting or repair occurs.
Verification does not certify a whole-circle bound. Non-complementary input with
verify=False still has the original caller precondition.

Verification requires an explicit FFT provider, matching the numerical module's
existing dependency boundary. Omitting it while verify=True raises a native
boundary ValueError before numerical work. No FFT is consumed when verify=False;
provider exceptions propagate and a rejected grid stops before complement.
Tests supply the same SciPy FFT as the unchanged reference. Production code does
not import SciPy. The native example below supplies NumPy FFT.

Only finite explicit sequences are accepted here. Symbolic/Shaped constructor
and factory branches, arbitrary foreign scalar/dtype behavior and high-degree
conditioning remain open. In particular, the source symbolic factory's behavior
of dropping the requested precision is not claimed implemented by this slice.

## Precision declaration

The direct constructor retains precision=1e-11; the polynomial factory retains
precision=0. For n coefficients, each source SU2 receives precision/(n+1), and
its Rz/Rx/Rz children receive that quantity divided by three. The zero-port
GlobalPhase retains its own default precision=1e-11. The cached
`signal_rotation_calls` exposes these existing resource identities and original
angle order without introducing a second rotation representation.

Tests compare those records to the unchanged source's actual decomposed SU2
children in creation-ID order, including GlobalPhase. Physical instructions are
unchanged when only the precision budget changes: the emitter uses continuous
rotations. No synthesis error guarantee, T count or hardware precision claim is
made from these declarations.

## Separate model ownership

`generalized_qsp_model.source_generalized_qsp_cost` receives coefficient count,
system width, nonnegative integer offset and four caller-priced component costs.
It does not compute roots, phases or physical gates. Its source-order calls are:

| Component | Repetitions | Bystander wires in the native width model |
|---|---:|---:|
| Signal SU2 | n | system width |
| Zero-controlled U | max(0,n-1-k) | 0; child includes signal |
| Bare U adjoint | max(0,k-n+1) | 1 signal |
| Positively controlled U adjoint | min(n-1,k) | 0; child includes signal |

Zero-multiplicity edges are omitted. The shared ComponentCost evaluator aggregates
child gate prices and sequential peak widths. Child costs must include their
workspace; assigning widths/prices remains a caller obligation. The bystander
policy is native width accounting, separately checked from upstream call keys.
The source top-level call graph uses an arbitrary SU2 key rather than pricing
actual numerical rotations or retaining each actual epsilon. We preserve that
separation: actual angle/precision records and declared component aggregation
are different views, neither a measured primitive histogram.

## Executable native example

```python
import numpy as np
import torch
import flagquantum as fq
from flagquantum.algorithms.primitives.generalized_qsp import GeneralizedQSP
from flagquantum.algorithms.primitives.generalized_qsp_model import source_generalized_qsp_cost
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost, GateCost, estimate_component_resources,
)
from flagquantum.compiler.registers.primitives.pauli_rotation import pauli_power_ir
from flagquantum.core.ir import Instruction

def fft(data):
    return np.asarray(np.fft.fft(data), dtype=np.complex128)

def unitary(signal, ports, workspace, value, adjoint):
    (target,) = ports["q"]
    gates = pauli_power_ir("z", target, -0.29 if adjoint else 0.29,
                           control=None if value is None else signal)
    if value == 0:
        flip = Instruction("x", (signal,))
        return (flip, *gates, flip)
    return gates

operation = GeneralizedQSP.from_qsp_polynomial(
    (("q", 1),), unitary, (0.25, 0.1, 0.2), negative_power=1,
    precision=2e-8, verify=True, fft=fft,
)
circuit = fq.Circuit(3, dtype=torch.complex128)
circuit.append(operation.make_on(signal=0, target=(1,)))
assert len(circuit.to_ir().instructions) == 25
assert torch.count_nonzero(circuit.state().reshape(4, 2)[:, 1]) == 0
assert operation.signal_rotation_calls[0][1].precision == (2e-8 / 4) / 3

# Explicit declared prices, not a synthesized or measured gate histogram.
model = source_generalized_qsp_cost(
    len(operation.p), 1, negative_power=operation.negative_power,
    signal_rotation=ComponentCost("SU2", 1, GateCost(rotation=3)),
    zero_control_unitary=ComponentCost("C[0]U", 2, GateCost(rotation=1)),
    unitary_adjoint=ComponentCost("U-adjoint", 1, GateCost(rotation=1)),
    controlled_adjoint=ComponentCost("CU-adjoint", 2, GateCost(rotation=1)),
)
estimate = estimate_component_resources(model)
assert estimate.gates.rotation == 11
assert estimate.peak_logical_qubits == 2
```

## Acceptance boundary

Thirty-five factory tests cover real/complex finite arrays, offsets 0/1/6,
verification on/off, original coefficients and full source circuit amplitudes
on four basis inputs plus a complex coherent input. An idle bit remains exactly
zero. All signal child angles, phase conventions and epsilon values match the
actual original SU2 decomposition. Changing precision leaves physical IR equal.
The source minimum-degree, root and grid failure boundaries are retained.

Twenty-two model tests compare original build_call_graph keys, insertion order
and multiplicities at n=2/3/6 and k=0/1/2/4/8. Shared aggregation is checked with
different prices for each child, active-call peak widths, omitted inactive
children and replacement child prices. Native finite input validation is
explicitly separate from source symbolic behavior.

The existing multi-register clean-work tests remain in the concentrated
regression. Arbitrary-U generic controlled leaf expansion, symbolic factory
behavior, physical synthesis and full-parent acceptance remain open; see
[GENERALIZED_QSP_EMISSION.md](GENERALIZED_QSP_EMISSION.md) and
[GENERALIZED_QSP_OPERATION.md](GENERALIZED_QSP_OPERATION.md).


## Recorded verification

Factory/precision commit `0e343b79` and declared-model commit `65931021` passed
57 new tests, 181 focused tests (5.78 s), 474 concentrated tests (4.13 s), and
64 integration tests (4.36 s). The focused/concentrated runs retained the 48
previously independently reproduced matmul warnings from the multi-register
fixture; the 57 new tests had no warnings. Both touched production modules
had 78 statements and 16 branches covered at 100%. Strict Python 3.12 typing
for four files, Black, Ruff, Algorithms scope, architecture, IR and public API
checks passed. Seven original source blobs matched the locked CSV; 17 unchanged
original tests passed (0.53 s) with six original marker warnings. The native
factory/model Markdown example was executed.

The required default run recorded 21507 passed, 300 skipped, 1037 warnings and
seven failures (126.56 s). All seven failure IDs matched the established
independent baseline; no new failures were observed. The maxfail-stopped run
is not a green or complete-suite result. Evidence is stored under
`ftqc-reference-env/g2-gqsp-factory-model-*`; publication is verified separately
in the external progress ledger.
