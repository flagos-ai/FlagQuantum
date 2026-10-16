# API Change: the von Neumann entropy output

## Status

Implemented for review on 2026-11-01 with explicit API-owner authorization for one
additive Stable Core root export, `fq.vn_entropy`, granted as part of the
measurement-parity family (`fq.variance`, `fq.vn_entropy`, `fq.mutual_info`,
`fq.purity`). It adds one root name, one field on `OutputRequest`, one property on
`ExecutionResult`, and one new arithmetic module. It changes no existing
signature, changes no serialized schema, and leaves `IR_VERSION` at `"1.0"`.

Under [`docs/development/PUBLIC_API_PROTECTION.md`](../development/PUBLIC_API_PROTECTION.md)
an additive stable API requires a concrete user journey, a reason the name belongs
in Stable Core rather than a namespace, typing and documentation, executable
behavior contracts, API-owner approval, a release-note entry, and the updated
machine-readable contract. Each is recorded below, and the design argument is in
[Proposal 069](../development/API_CHANGE_PROPOSAL_069_VN_ENTROPY_OUTPUT.md).

## Problem

The output vocabulary let a user ask *what state* a run left behind — proposal 068
added `fq.density_matrix` for exactly that — but not *how mixed* that state is.
The user had to diagonalise the matrix themselves, and three things went wrong at
once:

```python
import flagquantum as fq
import torch

circuit = fq.Circuit(2).h(0).cx(0, 1)
result = fq.run(circuit, outputs=fq.density_matrix(qubits=0))
eigenvalues = torch.linalg.eigvalsh(result.density_matrix[0]).real
positive = eigenvalues[eigenvalues > 1e-30]  # every user invents this floor
entropy = float(-(positive * torch.log(positive)).sum())
```

The floor is a numerical decision with no home, so each caller picks their own
and most of them pick silently. The reduction is written for a matrix and so
cannot run against an `mps` or `tensor_network` result, whose state is not a
matrix. And the whole thing has to be re-derived to answer the same question in
a different mode.

The PennyLane comparison is direct. `qml.vn_entropy(wires=[0])` is a measurement
a QNode returns alongside any other measurement, in every device:

```python
import pennylane as qml

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.vn_entropy(wires=[0])

circuit()  # 0.6931471805599454
```

FlagQuantum had no equivalent, which is the gap this change closes.

## Decision

### One new root export

```python
fq.vn_entropy(qubits, *, log_base=None, name=None) -> OutputRequest
```

```python
>>> import flagquantum as fq
>>> circuit = fq.Circuit(2).h(0).cx(0, 1)
>>> round(float(fq.run(circuit, outputs=fq.vn_entropy(0)).vn_entropy), 6)
0.693147
>>> round(float(fq.run(circuit, outputs=fq.vn_entropy([0], log_base=2)).vn_entropy), 6)
1.0
>>> round(float(fq.run(circuit, outputs=fq.vn_entropy([0, 1])).vn_entropy), 8)
6e-08
```

The last line is the whole Bell pair, which is pure: the exact answer is zero and
the reading is the `complex64` default precision. On a `complex128` circuit of the
same program it is `2.2204460492503126e-16`. That gap is the numerics boundary
stated below, visible in a two-line example.

The request is an ordinary member of the output family: it goes through
`fq.run(..., outputs=...)`, it survives `fq.plan(...)` and execution of a plan, it
carries an optional `name=` for selection when a program asks for more than one
reading, and it is dispatched by the same runtime path every other kind uses.

At least one qubit is required, and no observable is accepted — the entropy is a
property of a subsystem, not of a measured operator.

### One result property

```python
ExecutionResult.vn_entropy -> torch.Tensor
```

Shaped `(batch,)`, the same leading batch axis `fq.probabilities()` uses. It is a
property beside `.probabilities`, `.counts`, `.density_matrix`, and
`.expectations`, so it fails the same way they do when the requested output is
absent or ambiguous rather than returning `None`.

### One new arithmetic home

`flagquantum/simulation/entropy.py` owns the entropy arithmetic for the whole
package: `entropy_from_probabilities`, `entropy_from_density_matrix`,
`log_base_divisor`, `contiguous_ascending_run`, `mps_run_density_matrix` and
`mps_prefix_spectrum`. Every route in `flagquantum/runtime/measurements.py` calls
into it.

The placement follows the domain boundary rather than convenience.
`flagquantum/simulation/AGENTS.md` owns the numerical algorithms, and
`flagquantum/runtime/AGENTS.md` explicitly must not add numerical algorithms — so
Runtime chooses which route a mode takes and assembles the result, and Simulation
owns the logarithm. There is exactly one `p log p` sum in the package rather than
one per execution mode, which is what engineering decision principle 6 requires.

## The four measured properties

A signature check cannot tell a correct entropy from a plausible one, because
several wrong implementations return a number in the right range.
[`contracts/vn-entropy-output-contract.toml`](../../contracts/vn-entropy-output-contract.toml)
and [`tools/check_vn_entropy_output_contract.py`](../../tools/check_vn_entropy_output_contract.py)
record and re-measure four dimensions:

| Dimension | What would pass a signature check and fail here |
| --- | --- |
| definition | An implementation that ignores the selection and returns the whole state's entropy. On a globally pure program the entropy of a selection and of its complement are equal by mathematics, so the identity alone cannot tell which side was asked about. The gate therefore also runs two **mixed** rows and requires the two sides to separate by at least `state_separation_floor = 1e-6`. Measured on the recorded damped program: the damped qubit answers `0.19646697846469818`, the undamped qubit `2.2204460492503126e-16`. The gate also checks the dimensional bound `0 <= S <= k ln 2` and every recorded value. |
| agreement | An implementation that agrees with itself but not with the definition. The gate rebuilds the density matrix from the amplitudes, applies the damping channel by explicit Kraus contraction, traces out the complement with its own `einsum`, diagonalises with its own call, and sums its own `p log p`. None of that is the package's entropy code, so agreement is evidence about the module rather than a second reading of it. Tolerance `1e-12`; largest measured difference across the fourteen rows `7.7e-17`. |
| modes | An implementation that is correct on the dense path and wrong on a compressed one. All five execution modes run for every pure row — 60 route readings — and must agree to `1e-12`. Mixed rows are excluded, and the exclusion is *reported* rather than hidden: a noisy program refuses `mps`, `statevector` and `tensor_network`, so there is one route to compare there, and that refusal is itself a recorded row. |
| base | An implementation that accepts `log_base` and ignores it, or normalises it. Five recorded bases (natural, `2`, `10`, `e`, `0.5`) each equal the natural answer divided by the logarithm of the base, applied to the selection rather than to the whole state. |

Every refusal sentence the contract records is raised by a trigger the gate runs,
so a recorded message is the live one rather than a remembered one. The gate
checks both directions: a recorded refusal with no trigger fails, and a trigger
the contract does not record fails.

## The MPS route

This is the one place where implementing the item changed what the item was.

The plan recorded the behaviour as "the MPS executor reads the Schmidt values off
the canonical form directly — cheaper than the statevector route, dispatched by
mode". Measured against the code, the second half was not true as written:
`measurements.py` receives an `MPSState` object, and the shared
`_density_matrix_output` helper rebuilds a full matrix for *every* non-dense mode
before any kind-specific code runs. Reusing it would have kept the right number
while making the "cheaper route" claim false.

The route is therefore a routing decision, not a semantic one:

- Selection equal to the ascending run `0 .. k-1`: move the orthogonality centre
  to site `k-1` and read the Schmidt values of that cut. Their squares are the
  spectrum, so the sum runs over a rank bounded by the bond dimension rather than
  by `2 ** k`.
- Anything else — interior, non-contiguous, or descending: contract the named
  sites into a reduced matrix and diagonalise, costing `4 ** len(qubits)`.

Both routes are measured against the same recorded numbers, and the mode sweep
fails if they disagree. Their divergence on the recorded programs is exactly zero
in `complex128`, because they are two factorisations of one number rather than
two approximations of it.

One consequence is worth stating because the plan did not: the MPS route
**succeeds where the density-matrix route refuses**. `fq.density_matrix([0])` on
an eleven-qubit program is refused by the statevector amplitude ceiling;
`fq.vn_entropy([0])` on the same program is answered from the canonical form.
The contract's `[bound]` section records a probe at 12 qubits for that reason.

`contiguous_ascending_run` returning `None` is a routing predicate and not a
restriction on callers: a descending or non-contiguous selection still answers,
through the contract-and-diagonalise path. Order invariance is the semantics —
`fq.vn_entropy([0, 1])` and `fq.vn_entropy([1, 0])` name the same subsystem and
return the same number — checked against every mode and against PennyLane.

## `log_base`: the one deliberate divergence

The keyword name and its `None` default are copied verbatim from PennyLane, so a
user who knows one surface knows the other. The accepted range is where they
diverge, and the divergence is deliberate. Measured on PennyLane 0.45.1, whose
`pennylane/math/quantum.py` validates nothing and simply divides:

| `log_base` | PennyLane 0.45.1 | FlagQuantum |
| --- | --- | --- |
| `None` | `0.6931471805599454` (nats) | `0.6931471805599454` |
| `2` | `1.0000000000000002` | `1.0000000000000002` |
| `10` | `0.3010299956639812` | `0.3010299956639812` |
| `e` | `0.6931471805599454` | `0.6931471805599454` |
| `0.5` | `-1.0000000000000002` | `-1.0000000000000002` |
| `0` | natural answer, through a truthiness accident | `ValueError: log_base must be positive` |
| `-1` | `nan` | `ValueError: log_base must be positive` |
| `1` | `inf`, with `RuntimeWarning: divide by zero` | `ValueError: log_base must not be 1, which has no logarithm` |
| `inf` | `nan` | `ValueError: log_base must be finite` |
| `True` | `inf` (a boolean is `1`) | `TypeError: log_base must be a number` |
| `"2"` | `TypeError` from the division | `TypeError: log_base must be a number` |

`0` is accepted upstream through a truthiness accident rather than by intent, `1`
returns `inf` for a quantity that is mathematically undefined, and `True` is
silently read as a base. The repository's fail-closed principle says an input the
package cannot honour is refused at the earliest knowable stage with a sentence
naming the reason. `0.5` is deliberately **not** refused: it is a valid base, both
packages accept it, and both return the same sign.

The refusal happens in the request's constructor **and** the divisor is computed
again at measurement time, because a caller can construct an `OutputRequest`
directly and a plan can be deserialised. `log_base` is stored **unresolved** —
`None` stays `None` rather than being folded to a number — so the request stays a
faithful record of what was asked for and can be compared with the call that
produced the result. No normalisation stage exists.

## The zero-eigenvalue floor

`_PROBABILITY_FLOOR = 1e-30` is stated in `flagquantum/simulation/entropy.py`.
The sum is a floor, not a clamp: a probability below the floor contributes exactly
zero rather than being nudged to `1e-30`. Every recorded pure row contains exact
zeros — a Bell pair's reduced state has two — so the floor is load-bearing on all
of them, and the gate's independent reference uses its own `1e-30` floor so the
comparison is against the same stated convention rather than an unstated one.

## Numerics boundary

The contract's `mode_tolerance` and `independent_reference_tolerance` are both
`1e-12`, which is defensible only because every recorded baseline is
`complex128`:

| Reading | Largest measured difference |
| --- | --- |
| gate reference vs. the package, `complex128` | `7.7e-17` |
| across the five execution modes, `complex128` | `5.4e-15` |
| the same on a freshly rebuilt table | `1.915e-15` |
| the two mixed rows vs. the gate's reference | `0` (exactly) |
| `complex64` against `complex128`, same program | `2.184e-06` |
| MPS Schmidt shortcut against the dense matrix, `n=5`, cut 4 | `2.325e-06` |
| a batched MPS run against the dense matrix | `1.431e-06` |

`eps(complex64) = 1.1920928955078125e-07`, so the single-precision story is three
orders of magnitude looser and the contract does not claim it. A user who needs
the tight tolerance reads it off a `complex128` circuit, which is what the
recorded rows are.

## A pre-existing noise defect, recorded but not repaired

`amplitude_damping_channel` and `depolarizing_channel` narrow their rate through
`_real_dtype(_complex_dtype(None))`, which is `float32`, **even when
`dtype=torch.complex128` is passed**. So

```python
fq.amplitude_damping_channel(0.3)        # the rate is 0.30000001192092896
```

and a run using it reports `rho00 = 0.6500000112521603` where the exact answer is
`0.65`. A rate exactly representable in `float32` survives bit for bit, which is
why the contract's two mixed rows use `damping = 0.25`: the row then measures the
entropy, not the channel's rounding.

This change does not repair the defect — it is an existing behaviour of the noise
channels, unrelated to the entropy, and repairing it would silently change the
numbers every existing noisy run produces. It is recorded here because the
contract's mixed rows would otherwise look like a chosen precision floor rather
than a chosen rate. The gate constructs its channels with an explicit
`dtype=torch.complex128` and a comment saying that the channel's `complex64`
default would have hidden a `float32` error inside the tolerance.

## Compatibility

| Surface | Before | After |
| --- | --- | --- |
| `fq.__all__` | 37 names | 38 names |
| `fq.vn_entropy` | absent | `(qubits, *, log_base=None, name=None) -> OutputRequest` |
| `fq.OutputRequest` kinds | `counts`, `density_matrix`, `expectation`, `probabilities`, `samples` | plus `vn_entropy` |
| `OutputRequest` fields | `kind`, `qubits`, `observable`, `name` | plus `log_base=None` |
| `ExecutionResult` properties | `.probabilities`, `.counts`, `.density_matrix`, `.expectations` | plus `.vn_entropy` |
| `IR_VERSION` | `"1.0"` | `"1.0"` |
| serialized schemas | unchanged | unchanged |

`OutputRequest.log_base` defaults to `None` and is refused on every other kind, so
the new field is additive and cannot be set on a request that has no meaning for
it. An existing caller who never asks for an entropy sees no change at all.

The new name deliberately does not accept the deprecated `wires=` spelling its
five siblings carry. `counts`, `expectation`, `probabilities`, `samples` and
`density_matrix` each carry a `wires=` alias as recorded 0.3.x migration debt with
a removal version, and a name born after the migration started would be born
deprecated. This is the same decision proposal 068 recorded for
`fq.density_matrix`.

### A third firing of the construction-acceptance census

`contracts/construction-acceptance-contract.toml` records
`measured_root_export_count` as a live reading rather than a constant, precisely so
that a surface move is named instead of absorbed. This slice is the third movement:
it reads 38 after this PR and read 37 before it.

The census fired on this branch, in its own words:

```text
measured_root_export_count is stale: contract says 37, fq.__all__ has 38
```

The recorded number moves with it, and `[subject]` is left alone on purpose. The two
readings answer different questions: `[subject].not_covered` is the list of
*construction members* the acceptance contracts as absent, and a measurement output
kind is not a construction member, so adding `vn_entropy` there would be a false
entry. The count is the reading that is supposed to move.

That is also why updating it is not "updating a snapshot so a test passes": the gate
would fail again on the next surface move, and `tests/unit/test_construction_acceptance.py`
still asserts the reading equals `len(fq.__all__)` rather than a literal.

## Verification

```bash
python tools/check_vn_entropy_output_contract.py
# Von Neumann entropy output contract passed: 14 recorded values (12 pure, 2 mixed)
# reproduce an amplitude-side reference assembled in the gate and satisfy the identity on a
# selection and its complement; the pure rows agree across 5 execution modes (60 route
# readings) and the mixed rows separate their selection from their complement; 5 recorded
# bases agree with the natural entropy divided by their logarithm; 18 recorded refusals
# raise the sentence the contract states

python -m tools.public_api_snapshot
# public API migration baseline passed

python tools/docs_source_of_truth.py --check
# (no output: generated documentation is current)

python -m pytest tests/unit/test_vn_entropy_output_contract.py -q
# 15 passed

python -m pytest tests/unit/test_vn_entropy_output.py -q
# 7 passed

python tools/validate_required_checks.py
# validated 6 externally configured required checks
```

The scenario file is the readable half of the evidence: it builds the programs the
recorded numbers came from and checks each reading against a reduction and
diagonalisation written independently of the implementation.

## Refusal set

Eighteen refusal rows are recorded and each is triggered by the gate. Grouped by
what a caller did:

| The caller | The refusal |
| --- | --- |
| named no qubit | `vn_entropy output requires at least one qubit` |
| passed an observable | `vn_entropy output does not accept an observable` |
| repeated a qubit | `output qubits must be unique` |
| passed a negative, fractional, boolean, or non-integer qubit | `vn_entropy output qubit must be a non-negative integer` / `vn_entropy output qubit must be an integer` |
| named a qubit outside the program | `measurement qubits … are outside a …-qubit circuit` |
| asked for a base of `0`, a negative base, `1`, or a non-finite base | `log_base must be positive` / `log_base must not be 1, which has no logarithm` / `log_base must be finite` |
| asked for a boolean or string base | `log_base must be a number` |
| set `log_base` on a kind that has no base | `probabilities output does not accept a log_base` |
| attached a readout rule | the readout-confusion refusal, naming why an entropy cannot carry it |
| asked with `shots=` | `shots requires fq.samples(...) or fq.counts(...) output` |
| asked in `mode='stabilizer'` | `mode='stabilizer' samples measurement outcomes; it cannot serve measurement kind(s) vn_entropy` |
| asked a noisy program in a mode that cannot hold a mixed state | `stable noisy execution supports mode='auto' or mode='density_matrix'` |

A blank `name=` is refused by the shared output-name rule with `output name must
be a non-empty string`.
