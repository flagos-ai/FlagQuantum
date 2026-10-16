# API Change: the density-matrix output

## Status

Implemented for review on 2026-10-06 with explicit API-owner authorization for one
additive Stable Core root export, `fq.density_matrix`. It adds one root name, one
property on `ExecutionResult`, and one compatible implementation repair to an
existing method. It changes no existing signature, changes no serialized schema,
and leaves `IR_VERSION` at `"1.0"`.

Under [`docs/development/PUBLIC_API_PROTECTION.md`](../development/PUBLIC_API_PROTECTION.md)
an additive stable API requires a concrete user journey, a reason the name belongs
in Stable Core rather than a namespace, typing and documentation, executable
behavior contracts, API-owner approval, a release-note entry, and the updated
machine-readable contract. Each is recorded below, and the design argument is in
[Proposal 068](../development/API_CHANGE_PROPOSAL_068_DENSITY_MATRIX_OUTPUT.md).

## Problem

The output vocabulary lets a user ask *how* an execution is measured but not
*what state* it left behind. `fq.probabilities()`, `fq.counts()`, `fq.samples()`,
and `fq.expectation()` all describe outcomes; a user who wants the density matrix
of a subsystem has to leave the result object and reconstruct it by hand:

```python
import flagquantum as fq
import torch

circuit = fq.Circuit(3).ry(0, theta=0.7).ry(1, theta=1.1).ry(2, theta=0.4).cx(0, 1).cx(1, 2)
result = fq.run(circuit)
state = result.statevector  # the whole register, as a state vector
```

Two costs follow from that. The user has to know the internal basis order to
reshape the state and contract out the qubits they did not ask about, and a run
executed in `mps` or `density_matrix` mode does not hand out a state vector at
all, so the same script cannot be written once and run in every mode.

The PennyLane comparison is direct. `qml.density_matrix(wires=[0])` is a
measurement a QNode returns alongside any other measurement, and it is the
standard way to read a reduced state:

```python
import pennylane as qml

dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def circuit():
    qml.RY(0.7, wires=0)
    qml.RY(1.1, wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.density_matrix(wires=[0])

circuit().shape  # (2, 2)
```

FlagQuantum had no equivalent, which is the gap this change closes.

A second, independent defect surfaced while measuring the first. The circuit
object already had a `density_matrix()` method and a `noisy_density_matrix()`
method, and they disagreed about precision: `density_matrix()` read the
circuit's dtype while `noisy_density_matrix()` silently built at the
process-wide runtime default. On a `complex128` circuit the two paths differed by
the `complex64` rounding floor, and an operator at the circuit's dtype could not
be contracted against the noisy result at all:

```python
>>> import flagquantum as fq, torch
>>> circuit = fq.Circuit(2, dtype=torch.complex128).h(0).cx(0, 1)
>>> circuit.density_matrix().dtype
torch.complex128
>>> circuit.noisy_density_matrix(fq.NoiseModel()).dtype  # before the repair
torch.complex64
```

That is a compatible implementation change to an existing method rather than an
additive API, so it requires no proposal of its own; it is recorded here because
it is part of the same user journey and it is what makes the module docstring of
[`examples/algorithms/error_mitigation.py`](../../examples/algorithms/error_mitigation.py)
honest. That file's comment recorded the wound verbatim:

```text
# Circuit.density_matrix() takes no dtype and returns the runtime's default
# precision, so a distance taken from it would report the complex64 rounding
# floor as the extrapolation's error.
```

## Decision

### One new root export

```python
fq.density_matrix(qubits=None, *, name=None) -> OutputRequest
```

```python
>>> import flagquantum as fq
>>> circuit = fq.Circuit(2).h(0).cx(0, 1)
>>> fq.run(circuit, outputs=fq.density_matrix(qubits=0)).density_matrix.shape
torch.Size([1, 2, 2])
>>> fq.run(circuit, outputs=fq.density_matrix()).density_matrix.shape
torch.Size([1, 4, 4])
```

The request is an ordinary member of the output family: it goes through
`fq.run(..., outputs=...)`, it survives `fq.plan(...)` and execution of a plan, it
carries an optional `name=` for selection when a program asks for more than one
matrix, and it is dispatched by the same runtime path every other kind uses.

### One result property

```python
ExecutionResult.density_matrix -> torch.Tensor
```

It is a property beside `.probabilities`, `.counts`, and `.expectations`, so it
fails the same way they do when the requested output is absent or ambiguous
rather than returning `None`.

### One compatible repair

`Circuit.noisy_density_matrix` now forwards the circuit's dtype to the noisy
executor, matching `Circuit.density_matrix`, `Circuit.state`, and every other
circuit-level read.

## The three measured properties

A signature check cannot tell a correct matrix from a plausible one, so the
contract records what the matrix *is* and the gate re-measures it on every run.
[`contracts/density-matrix-output-contract.toml`](../../contracts/density-matrix-output-contract.toml)
and [`tools/check_density_matrix_output_contract.py`](../../tools/check_density_matrix_output_contract.py)
cover four dimensions:

| Dimension | What would pass a signature check and fail here |
| --- | --- |
| trace | A marginal instead of a trace: summing a row or column index instead of contracting the row index against the column index returns a matrix whose trace is the whole state's purity, which is one for a pure state and therefore passes a Bell-pair check by luck. The gate therefore also runs a mixed program and refuses to proceed if that probe turns out to be pure. |
| agreement | A reduction that does not reproduce the observable path: `Tr(O rho_S)` is read off the reduced matrix and compared with what `fq.expectation(...)` reports for the same operator, which the gate builds by explicit Kronecker products rather than through the helper the reduction uses. |
| permutation | A reduction that answers in ascending qubit order regardless of the selection. The gate builds the basis permutation from the naming and applies it, so an implementation that ignored the caller's order fails instead of passing as a second reading. |
| cost | An unbounded statevector route. Building a matrix from amplitudes writes `4 ** n` entries, the square of the state it is built from, so the statevector route refuses beyond ten qubits and names `mode='density_matrix'` as the cheaper route. |

Every refusal sentence the contract records is raised by a trigger the gate runs,
so a recorded message is the live one rather than a remembered one.

## The ordering decision

`qml.density_matrix(wires=[1, 0])` returns the same numbers as `wires=[0, 1]` with
rows and columns 1 and 2 exchanged: PennyLane honours the order the caller named.
FlagQuantum honours it too, and the reason is not parity for its own sake.
`fq.probabilities(qubits=(1, 0))` already answers in the caller's order, so a
matrix that answered in ascending order would be the one output kind in the
family whose indexing did not match how it was asked for.

Measured against PennyLane 0.45.1 on `default.qubit` with the same gate list, the
largest absolute difference over the whole matrix is `1.4e-16` across
`(0,)`, `(1,)`, `(2,)`, `(0,1)`, `(1,0)`, `(2,1)`, and `(0,2)`.

## Batch axis

`ExecutionResult.density_matrix` is a batched tensor shaped
`(bsz, 2 ** k, 2 ** k)`, the same convention `fq.probabilities()` uses for its
`(bsz, 2 ** k)` tensor and the same convention `Circuit.density_matrix()` already
returns. PennyLane returns a bare `ndarray` with no batch axis. That is the one
deliberate difference between the two surfaces, and it is the difference every
other FlagQuantum output already has.

## Compatibility

| Surface | Before | After |
| --- | --- | --- |
| `fq.__all__` | 36 names | 37 names |
| `fq.density_matrix` | absent | `(qubits=None, *, name=None) -> OutputRequest` |
| `fq.OutputRequest` kinds | `counts`, `expectation`, `probabilities`, `samples` | plus `density_matrix` |
| `ExecutionResult` properties | `.probabilities`, `.counts`, `.expectations` | plus `.density_matrix` |
| `Circuit.noisy_density_matrix` | runtime default dtype | the circuit's dtype |
| `IR_VERSION` | `"1.0"` | `"1.0"` |
| serialized schemas | unchanged | unchanged |

An existing caller who never asks for a density matrix sees no change other than
the dtype repair, which makes a previously inconsistent read consistent. A caller
who relied on the noisy matrix being `complex64` while their circuit was
`complex128` was relying on a defect.

The new name deliberately does not accept the deprecated `wires=` spelling its
four siblings accept: `counts`, `expectation`, `probabilities`, and `samples`
each carry one as recorded 0.3.x migration debt with a removal version, and a
name born after the migration started would be born deprecated. See §9 of the
proposal for why this is the one place the new signature diverges from the
family.

## Verification

```bash
python tools/check_density_matrix_output_contract.py
# Density-matrix output contract passed: 5 recorded matrices trace to one, agree with the
# observable path, reproduce under the caller's qubit order as a basis permutation and across
# 4 execution modes; 6 recorded refusals raise the sentence the contract states

python -m tools.public_api_snapshot
# public API migration baseline passed

python -m pytest tests/unit/test_density_matrix_output.py \
    tests/unit/test_density_matrix_output_contract.py -q
# 37 passed

python tools/validate_required_checks.py
# validated 6 externally configured required checks
```
