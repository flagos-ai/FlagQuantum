# API Change Proposal 068: The density-matrix output

## Status

**Proposed.** This proposal adds one additive Stable Core root export,
`fq.density_matrix(qubits=None, *, name=None)`, one property on
`ExecutionResult`, and one compatible repair to `Circuit.noisy_density_matrix`.
The authorization record is
[`FQ-DENSITY-MATRIX-OUTPUT-20261006.md`](../api-changes/FQ-DENSITY-MATRIX-OUTPUT-20261006.md).

It proposes one public name and one result property. It changes no existing
signature, no default, no serialized schema, and no `IR_VERSION`; the repair is a
compatible implementation change to an existing method rather than a signature
change.

## 1. The gap, measured

The output vocabulary is a family of *outcome* descriptions. Read from the live
package:

```python
>>> import flagquantum as fq
>>> [name for name in fq.__all__ if name in
...  ("counts", "density_matrix", "expectation", "probabilities", "samples")]
['counts', 'density_matrix', 'expectation', 'probabilities', 'samples']
```

Four of those five answer "what outcomes did this program produce, and how
often". None answers "what state did it leave on these qubits". A user who wants
a subsystem's state has to take `result.statevector`, know the basis order, and
contract out the qubits they did not ask about by hand.

Three consequences, in the order a user meets them:

1. **The same script is not portable across modes.** `result.statevector` exists
   in `statevector` and `mps` mode and not in `density_matrix` mode, so reading a
   reduced state is the one operation that cannot be written once.
2. **The basis order is an implementation detail the user must know.** Nothing in
   the public surface says whether the register index is big-endian or
   little-endian, and nothing says what the reduced matrix's index means.
3. **There is no place to put the answer.** `fq.run(..., outputs=...)` is the one
   route every execution takes, and it had no member for this.

PennyLane has the operation a user reaches for:

```python
import pennylane as qml

dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def circuit():
    qml.RY(0.7, wires=0)
    qml.RY(1.1, wires=1)
    qml.RY(0.4, wires=2)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    return qml.density_matrix(wires=[0])

circuit().shape  # (2, 2)
```

`qml.density_matrix` is a measurement the QNode returns, is composable with
every other measurement, and is one of the two ways PennyLane exposes a state
(the other being `qml.state()` for the whole register). FlagQuantum's
`result.statevector` is the analogue of `qml.state()`; this change adds the
analogue of `qml.density_matrix`.

## 2. The second, independent defect

`Circuit` already had both methods, and they disagreed about precision:

```python
>>> import flagquantum as fq, torch
>>> circuit = fq.Circuit(2, dtype=torch.complex128).h(0).cx(0, 1)
>>> circuit.density_matrix().dtype
torch.complex128
>>> circuit.noisy_density_matrix(fq.NoiseModel()).dtype  # before the repair
torch.complex64
```

`density_matrix()` reads the circuit's dtype; `noisy_density_matrix()` built at
the process-wide runtime default and dropped the circuit's. On a `complex128`
circuit the two paths disagreed by the `complex64` rounding floor
(`5.96e-08` against a `complex128` reference along the same code path), and an
operator at the circuit's dtype could not be contracted against the noisy result
at all:

```python
RuntimeError: expected scalar type ComplexFloat but found ComplexDouble
```

The wound was already recorded in the repository, in
[`examples/algorithms/error_mitigation.py`](../../examples/algorithms/error_mitigation.py):

```text
# Circuit.density_matrix() takes no dtype and returns the runtime's default
# precision, so a distance taken from it would report the complex64 rounding
# floor as the extrapolation's error.
```

That comment is why the example measures its reference through `run_zne(...,
dtype=DTYPE)` instead of through the circuit method, which is a workaround a user
should not have to invent. The repair is a compatible implementation change to an
existing method, so under
[`PUBLIC_API_PROTECTION.md`](../development/PUBLIC_API_PROTECTION.md) it needs no
proposal of its own; it is recorded in the same change because it is the same user
journey, and the stale comment is corrected with it.

## 3. Why this is additive rather than a repair of `Circuit.density_matrix()`

`Circuit.density_matrix()` already returns a matrix, so the cheapest change would
have been to teach the circuit object a `qubits=` argument. Three reasons not to:

1. **It would put a second execution route on the circuit object.** `Circuit.run`
   and `fq.run` are the routes that honor `ExecutionOptions`, noise, planning, and
   distributed execution. A `qubits=` argument on a circuit method would have to
   reimplement the reduction locally, and the two routes would then be able to
   disagree about what a reduction is.
2. **It would not compose.** A user who wants a matrix *and* counts from one
   execution has to ask for both in `outputs=`; a circuit method can only answer
   one question per call, and cannot be planned.
3. **It would name an operation twice.** The output family already has exactly
   one member per question. The question "the state on these qubits" belongs in
   the family next to "the state on these qubits as probabilities", not on the
   circuit object.

The circuit methods are unchanged in signature and keep working;
`Circuit.density_matrix()` keeps returning the whole register, which is the
correct answer to the question it asks.

## 4. Why Stable Core rather than a namespace or experimental

`PUBLIC_API_PROTECTION.md`'s second requirement is a reason the name belongs in
Stable Core. Three arguments:

- **It is a member of an existing stable family.** `fq.probabilities`,
  `fq.counts`, `fq.samples`, and `fq.expectation` are all stable root exports
  whose only argument is a qubit selection. Reaching a density matrix through a
  namespace would mean the same argument has two homes depending on the output
  kind, which is exactly the inconsistency the naming review rejects.
- **Its contract is expressible and stable.** The matrix is defined by
  mathematics, not by an implementation: `Tr(O rho_S) == <O>`, trace one, index
  order equal to the order the caller named. Section 6 records the gate that
  measures all four, so the meaning does not drift.
- **It is not provisional.** There is no experimental variant to graduate from
  and no second design competing with it. An experimental namespace would be a
  transitional label, which the naming review rejects when a precise domain term
  exists.

The name is `density_matrix`. It states the mathematical object; it is not
relative, not maturity-based, and carries no version suffix.

## 5. The request, and the three dimensions the contract measures

```python
fq.density_matrix(qubits=None, *, name=None) -> OutputRequest
```

```python
>>> import flagquantum as fq
>>> circuit = fq.Circuit(2).h(0).cx(0, 1)
>>> result = fq.run(circuit, outputs=fq.density_matrix(qubits=0))
>>> result.density_matrix.shape
torch.Size([1, 2, 2])
```

A signature check cannot tell a correct matrix from a plausible one. Writing the
reduction three wrong ways still produces a `(1, 2, 2)` complex tensor:

| Wrong implementation | What it returns | Caught by |
| --- | --- | --- |
| sum a row index instead of contracting it against the column index | a marginal whose trace is the state's purity, which is one for any pure state | the trace measurement, run on a mixed program |
| contract correctly but renormalize after every qubit removed | a trace-one matrix that is not the reduced state | the trace measurement on the mixed program, and the agreement measurement |
| contract correctly but always answer in ascending qubit order | the right matrix under the wrong index meaning | the permutation measurement |

The contract therefore records four dimensions and
[`tools/check_density_matrix_output_contract.py`](../../tools/check_density_matrix_output_contract.py)
re-measures them on every run:

| Dimension | Measurement |
| --- | --- |
| trace | The trace of every recorded matrix is one, and the gate refuses to proceed if its deliberately mixed probe turns out to be pure, so it cannot report a check it did not perform. |
| agreement | `Tr(O rho_S)` read off the reduced matrix is compared with what `fq.expectation(...)` reports for the same operator, which the gate lifts by explicit Kronecker products rather than through the helper the reduction uses. |
| permutation | The matrix for `(1, 0)` is compared with the basis permutation of the matrix for `(0, 1)`, so an implementation that ignored the order fails rather than passing as a second reading. |
| routes and refusals | Every execution mode named in the contract answers the same matrix, and every refusal sentence the contract records is raised by a trigger the gate runs. |

The measured route agreement on the three-qubit chain reduced to `(0, 1)`,
against the `statevector` answer:

```text
statevector      max|diff| = 0.000e+00
density_matrix   max|diff| = 2.220e-16
mps              max|diff| = 6.661e-16
auto             max|diff| = 0.000e+00
```

## 6. The ordering decision, and the PennyLane comparison

`qml.density_matrix(wires=[1, 0])` returns the same numbers as `wires=[0, 1]` with
rows and columns 1 and 2 exchanged: PennyLane honours the order the caller named.
Measured on `RY(0.7) RY(1.1) RY(0.4) CNOT(0,1) CNOT(1,2)`, the real part of the
`(0, 1)` matrix is

```text
[[0.641342 0.153123 0.143533 0.091166]
 [0.153123 0.241079 0.034269 0.143533]
 [0.143533 0.034269 0.032123 0.020403]
 [0.091166 0.143533 0.020403 0.085456]]
```

and the `(1, 0)` matrix is the same table with rows and columns 1 and 2 exchanged,
so the `0.153123` at `[0, 1]` moves to `[0, 2]`. PennyLane 0.45.1 prints the same
two tables entry for entry:

```text
wires=[0, 1]  [0,1] = 0.15312322012762453   [0,2] = 0.14353288608699652
wires=[1, 0]  [0,1] = 0.14353288608699652   [0,2] = 0.15312322012762453
```

FlagQuantum honours the order for a reason that is independent of parity:
`fq.probabilities(qubits=(1, 0))` already answers in the caller's order, so a
matrix that always answered in ascending order would be the one output kind in
the family whose meaning did not match how it was asked for.

The gate carries the comparison as a recorded number rather than a memory:
PennyLane `0.45.1` on `default.qubit` with the same gate list agrees with
FlagQuantum to `1.4e-16` across `(0,)`, `(1,)`, `(2,)`, `(0,1)`, `(1,0)`,
`(2,1)`, and `(0,2)`.

## 7. The one deliberate difference: the batch axis

`ExecutionResult.density_matrix` is a batched tensor shaped
`(bsz, 2 ** k, 2 ** k)`. PennyLane returns a bare `ndarray` with no batch axis.

The batch axis is kept because it is what every other FlagQuantum output already
has: `fq.probabilities()` returns `(bsz, 2 ** k)` and `Circuit.density_matrix()`
already returns `(bsz, 2 ** n, 2 ** n)`. A matrix output that dropped it would be
the one member of the family that could not be used in a training loop, which is
the primary interface this repository exists to serve.

## 8. The statevector bound

The reduction is arithmetic on amplitudes. The statevector route holds `2 ** n`
amplitudes and the matrix is `4 ** n` entries, the square of the state, so the
cost of asking grows quadratically with the program width. The density-matrix
route has already paid for the matrix and can reduce as far as asked; the
statevector route refuses above ten qubits and names the cheaper route in the
message.

This is `fail closed` rather than a silent downgrade: the refusal is a
`ValueError` at the point of the request, and the message contains the exact call
that is not bounded:

```text
a 12-qubit density-matrix output built from a statevector writes 16777216 entries,
which exceeds the supported limit of 10 qubits; run the program with
options=fq.ExecutionOptions(mode='density_matrix') to ask for the smaller reduction
instead
```

Measured: the same 12-qubit program succeeds in `density_matrix` mode and returns
the `(1, 2, 2)` reduction it was asked for.

## 9. What the change does not do

- **It does not add a readout-aware matrix.** A `NoiseModel` with a readout rule
  is refused by name: readout confusion is applied to *outcomes after*
  measurement and is not part of `rho`, so a matrix that folded it in would
  report a state the engine never prepared. PennyLane cannot express the
  combination either; mid-circuit measurement with `qml.density_matrix` is
  refused on `0.45.1`.
- **It does not change `Circuit.density_matrix()`'s signature.** It keeps
  answering for the whole register.
- **It does not touch the empty-selection convention.** `fq.probabilities(())`
  and `fq.counts(())` already mean "no selection", i.e. every qubit, and
  `fq.density_matrix(())` reads the same way. Pinning it here was considered and
  rejected: changing one member of the family while the others stood would be a
  worse inconsistency than the one it fixed.
- **It does not add a root export for the reduction itself.** The reduction is
  an argument of the output kind, not a second operation.
- **It does not add the `wires=` alias its four siblings accept.** `fq.counts`,
  `fq.expectation`, `fq.probabilities`, and `fq.samples` each take a deprecated
  `wires=` spelling that forwards to `qubits=` and warns. This new name takes
  only `qubits=`. Two reasons, and the second is the deciding one. The 0.3.x
  migration is removing `wires` from the FlagQuantum surface in 0.4.0, so a name
  introduced now would be born deprecated. And a root export declared today is a
  frozen stable name under rule 8, so adding `wires=` here would write a
  word the migration exists to delete into a contract that cannot drop it
  without a second breaking change. The four existing aliases are a recorded
  migration debt with a removal version; this one would be new debt with none.

  This is a deliberate divergence from the sibling signature, and it is why
  `contracts/observable-outputs-v1-candidate.json` records
  `(qubits=..., *, name=...) -> OutputRequest`. Removing the alias after the
  next migration would be a breaking change to a stable signature.

## 10. Compatibility

| Surface | Before | After |
| --- | --- | --- |
| `fq.__all__` | 36 names | 37 names |
| `fq.density_matrix` | absent | `(qubits=None, *, name=None) -> OutputRequest` |
| `fq.OutputRequest` kinds | `counts`, `expectation`, `probabilities`, `samples` | plus `density_matrix` |
| `ExecutionResult` properties | `.probabilities`, `.counts`, `.expectations` | plus `.density_matrix` |
| `Circuit.noisy_density_matrix` | runtime default dtype | the circuit's dtype |
| `docs/public_api_v1.json` `stable_exports` | 36 | 37 |
| `contracts/public-api-v1-candidate.json` `rules.root_export_budget` | 36 | 37 |
| `IR_VERSION` | `"1.0"` | `"1.0"` |
| serialized schemas | unchanged | unchanged |

The budget bump is the honest part of the change: an addition creates a
maintenance obligation, and the machine-readable contract records it rather than
absorbing it silently. No test assertion was relaxed to make this pass; the
count in `tests/unit/test_public_api_candidate.py` moved with the budget, and the
reviewed signature in
`contracts/observable-outputs-v1-candidate.json` is what
`tools/public_api_snapshot.py` checks the live export against.

## 11. Open questions

1. **Should the statevector bound be configurable?** Today it is a constant. A
   user with 64 GB can reduce further than a user with 8 GB, and the refusal does
   not say which. An explicit option would put a memory policy on the user, which
   the ten-qubit constant does not. Left as a constant until a user journey
   demonstrates otherwise.
2. **Should `Circuit.density_matrix()` gain the same `qubits=` argument as a
   convenience?** It would duplicate the request surface. Left out.
3. **Should the reduced matrix be available as a plan-level output for
   distributed execution?** It is, through `fq.plan(..., outputs=...)`, which was
   measured to round-trip. Whether a sharded execution can answer it without
   gathering is a separate question about the sharding layout rather than about
   this API.

## Verification

```bash
python tools/check_density_matrix_output_contract.py
# Density-matrix output contract passed: 5 recorded matrices trace to one, agree with the
# observable path, reproduce under the caller's qubit order as a basis permutation and across
# 4 execution modes; 6 recorded refusals raise the sentence the contract states

python -m pytest tests/unit/test_density_matrix_output.py \
    tests/unit/test_density_matrix_output_contract.py -q
# 37 passed

python -m tools.public_api_snapshot
# public API migration baseline passed

python -m pytest tests/api_contract/test_observable_outputs.py -q
# 61 passed

python tools/validate_required_checks.py
# validated 6 externally configured required checks
```

Every number in this proposal is produced by those commands at the `N2-5` base
commit and is reconciled by the gate on every run; none is transcribed by hand.
