# API Change Proposal 069: The von Neumann entropy output

## Status

**Proposed.** This proposal adds one additive Stable Core root export,
`fq.vn_entropy(qubits, *, log_base=None, name=None)`, and one property on
`ExecutionResult`, `ExecutionResult.vn_entropy`. The authorization record is
[`FQ-VN-ENTROPY-OUTPUT-20261101.md`](../api-changes/FQ-VN-ENTROPY-OUTPUT-20261101.md).

It proposes one public name and one result property. It changes no existing
signature, no default, no serialized schema, and no `IR_VERSION`.

It is the second PR of the measurement-parity family
(`fq.variance`, `fq.vn_entropy`, `fq.mutual_info`, `fq.purity`), which was
authorized as one grant. Each member of that family still lands as its own
change with its own proposal and its own contract, because each answers a
different question and each carries its own refusal set.

## 1. The gap, measured

The output vocabulary describes *outcomes*. Read from the live package before
this change:

```python
>>> import flagquantum as fq
>>> [name for name in fq.__all__ if name in
...  ("counts", "density_matrix", "expectation", "probabilities", "samples")]
['counts', 'density_matrix', 'expectation', 'probabilities', 'samples']
```

`fq.density_matrix` was added by
[Proposal 068](API_CHANGE_PROPOSAL_068_DENSITY_MATRIX_OUTPUT.md) and answers
"what state is on these qubits". It does not answer "how mixed is that state".
A user who wants the entropy has to take the matrix and diagonalise it
themselves, and — the part that matters — has to know that the eigenvalues of a
reduced density matrix are the thing to sum, that a zero eigenvalue must be
skipped rather than logged, and which basis order the matrix's indices use:

```python
import flagquantum as fq
import torch

circuit = fq.Circuit(2).h(0).cx(0, 1)
result = fq.run(circuit, outputs=fq.density_matrix(qubits=0))
eigenvalues = torch.linalg.eigvalsh(result.density_matrix[0]).real
positive = eigenvalues[eigenvalues > 1e-30]  # and the floor is the user's problem
entropy = float(-(positive * torch.log(positive)).sum())
```

Three consequences, in the order a user meets them:

1. **Every mode is a different transcription of the same arithmetic.** A
   `density_matrix` result hands out a matrix, an `mps` result hands out a
   matrix-product state whose canonical form *is* a spectrum, and a
   `tensor_network` result hands out contractible tensors. The user who writes
   the reduction above can only run it in one of them.
2. **The zero-eigenvalue floor is a numerical decision with no home.** A pure
   state's reduced matrix has exact zeros on the small side; `0 * log(0)` is
   `nan`, and every user invents their own floor, most of them silently.
3. **There is no place to put the answer.** `fq.run(..., outputs=...)` is the one
   route every execution takes, and it had no member for this.

PennyLane has the operation a user reaches for. Measured on PennyLane 0.45.1:

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

`qml.vn_entropy`, `qml.mutual_info`, `qml.purity` and `qml.variance` are
measurements a QNode returns alongside any other measurement. FlagQuantum had no
equivalent, which is the gap this change closes for the entropy member.

## 2. What is being added, and what is not

Nothing in this change is a new numerical core. The arithmetic is a partial
trace, a diagonalisation, and a sum of `p log p`. What the change adds is the
*placement*: one home for that arithmetic, one public name for asking, and one
place on the result for the answer.

### One new root export

```python
fq.vn_entropy(qubits, *, log_base=None, name=None) -> OutputRequest
```

```python
>>> import flagquantum as fq
>>> circuit = fq.Circuit(2).h(0).cx(0, 1)
>>> fq.run(circuit, outputs=fq.vn_entropy(0)).vn_entropy.shape
torch.Size([1])
>>> fq.run(circuit, outputs=fq.vn_entropy([0], log_base=2)).vn_entropy.shape
torch.Size([1])
```

The request is an ordinary member of the output family: it goes through
`fq.run(..., outputs=...)`, it survives `fq.plan(...)` and execution of a plan, it
carries an optional `name=` for selection when a program asks for more than one
reading, and it is dispatched by the same runtime path every other kind uses.

### One result property

```python
ExecutionResult.vn_entropy -> torch.Tensor
```

It is a property beside `.probabilities`, `.counts`, `.density_matrix`, and
`.expectations`, so it fails the same way they do when the requested output is
absent or ambiguous rather than returning `None`.

### One new arithmetic home

```python
flagquantum/simulation/entropy.py
```

Engineering decision principle 6 forbids a second source of truth, and § Current
Strategic Priority clause 2 admits a new horizontal abstraction only when it
**replaces** an existing implementation. Neither condition is met by returning a
matrix and letting each mode reduce it separately: the same `p log p` sum with
the same floor would then exist once per mode.

The module is therefore the single home of the entropy arithmetic —
`entropy_from_probabilities`, `entropy_from_density_matrix`,
`log_base_divisor`, and the two MPS spectrum readers — and every route in
`flagquantum/runtime/measurements.py` calls into it. Runtime keeps its own
responsibility (choosing which route a mode takes, and assembling the result)
and owns no logarithm. This is the *replacement* the clause asks for in the
sense that matters here: before this change there was no entropy implementation
at all, and after it there is exactly one, rather than one per execution mode.

## 3. The three dimensions the contract measures

A signature check cannot tell a correct entropy from a plausible one, because
several wrong implementations return a number in the right range. The contract
[`contracts/vn-entropy-output-contract.toml`](../../contracts/vn-entropy-output-contract.toml)
and its gate
[`tools/check_vn_entropy_output_contract.py`](../../tools/check_vn_entropy_output_contract.py)
record and re-measure four dimensions.

| Dimension | What would pass a signature check and fail here |
| --- | --- |
| definition | An implementation that ignores the selection and returns the whole state's entropy. On a globally pure program the entropy of a subsystem and of its complement are equal by mathematics, so the identity alone cannot tell which side was asked about; the gate therefore also runs a **mixed** program, where the two sides differ, and requires them to separate by at least `state_separation_floor = 1e-6`. Measured on the recorded damped program: the damped qubit answers `0.19646697846469818` and the undamped qubit `2.2204460492503126e-16`. The gate also checks the dimensional bound `0 <= S <= k ln 2` and each recorded value. |
| agreement | An implementation that agrees with itself but not with the definition. The gate rebuilds the density matrix from the amplitudes, applies the damping channel by explicit Kraus contraction, traces out the complement with its own `einsum`, diagonalises with its own call, and sums its own `p log p`. None of that is the package's entropy code, so agreement is evidence about the module rather than a second reading of it. Tolerance `1e-12`; the largest measured difference across the fourteen rows is `7.7e-17`. |
| modes | An implementation that is correct on the dense path and wrong on a compressed one. All five execution modes are run for every pure row (60 route readings) and must agree to `1e-12`. Mixed rows are excluded, and the exclusion is reported rather than hidden: a noisy program refuses `mps`, `statevector` and `tensor_network` with `ValidationError: stable noisy execution supports mode='auto' or mode='density_matrix'`, so there is only one route to compare there, and the contract carries that refusal as a recorded row. |
| base | An implementation that accepts `log_base` and ignores it, or that normalises it. Five recorded bases (natural, `2`, `10`, `e`, `0.5`) each have to equal the natural answer divided by the logarithm of the base, and the base has to be applied to the selection rather than to the whole state. |

Every refusal sentence the contract records is raised by a trigger the gate
runs, so a recorded message is the live one rather than a remembered one, and
the gate checks both directions: a recorded refusal with no trigger fails, and a
trigger the contract does not record fails.

## 4. The MPS route, and why the plan's line needed a second look

The plan recorded this item as "the MPS executor reads the Schmidt values off the
canonical form directly — cheaper than the statevector route, dispatched by
mode". Measured against the code, the second half of that line was not true as
written: `measurements.py` receives an `MPSState` object, and the shared
`_density_matrix_output` helper rebuilds a full matrix for every non-dense mode
before any kind-specific code runs. Reusing that helper would have made the
"cheaper route" claim false while still returning the right number, which is the
worst of both outcomes.

The route was therefore built rather than assumed, and it is a *routing*
decision rather than a semantic one:

- If the selection is the ascending run `0 .. k-1`, the centre is moved to site
  `k-1` and the Schmidt values of that cut are read directly. Their squares are
  the spectrum, so the entropy is one `p log p` sum over a rank that is bounded
  by the bond dimension rather than by `2 ** k`.
- Otherwise — a non-contiguous, descending, or interior selection — the named
  sites are contracted into a reduced matrix and diagonalised, which costs
  `4 ** len(qubits)`.

Both routes are measured against the same recorded numbers, and the gate's mode
sweep fails if they disagree. The measured divergence between the two routes on
the recorded programs is exactly zero in `complex128`, because they are two
factorisations of the same number, not two approximations of it.

A separate behaviour is worth stating because the plan did not: the MPS route
**succeeds where the density-matrix route refuses**. `fq.density_matrix([0])` on
an eleven-qubit program is refused by the statevector amplitude ceiling, and
`fq.vn_entropy([0])` on the same program is answered from the canonical form.
The contract's `[bound]` section records a probe at 12 qubits for this reason,
and the gate measures it.

### A run is a Schmidt cut only when it lines up

`contiguous_ascending_run` returns `None` for anything else, and that is a
routing predicate, not a restriction on what a caller may ask for: a descending
or non-contiguous selection still answers, through the contract-and-diagonalise
path. Order invariance *is* the semantics — `fq.vn_entropy([0, 1])` and
`fq.vn_entropy([1, 0])` are the same subsystem and therefore the same number —
and the gate checks that against every mode and against PennyLane.

## 5. `log_base`: the name, the type, and the one deliberate divergence

The keyword name and its `None` default are copied verbatim from PennyLane, so a
user who knows one surface knows the other:

```python
qml.vn_entropy(wires=[0], log_base=2)  # PennyLane
fq.vn_entropy([0], log_base=2)         # FlagQuantum
```

The type and the accepted range are where the two diverge, and the divergence is
deliberate. Measured on PennyLane 0.45.1, `pennylane/math/quantum.py` validates
nothing and simply divides:

| `log_base` | PennyLane 0.45.1 | FlagQuantum |
| --- | --- | --- |
| `None` | `0.6931471805599454` (nats) | `0.6931471805599454` |
| `2` | `1.0000000000000002` | `1.0000000000000002` |
| `10` | `0.3010299956639812` | `0.3010299956639812` |
| `e` | `0.6931471805599454` | `0.6931471805599454` |
| `0.5` | `-1.0000000000000002` | `-1.0000000000000002` |
| `0` | natural answer, by a truthiness accident | `ValueError: log_base must be positive` |
| `-1` | `nan` | `ValueError: log_base must be positive` |
| `1` | `inf` (with `RuntimeWarning: divide by zero`) | `ValueError: log_base must not be 1, which has no logarithm` |
| `inf` | `nan` | `ValueError: log_base must be finite` |
| `True` | `inf` (a boolean is `1`) | `TypeError: log_base must be a number` |
| `"2"` | `TypeError` from the division | `TypeError: log_base must be a number` |

The reasoning is not "FlagQuantum is stricter". It is that `0` is *accepted* by
PennyLane through a truthiness accident rather than by intent, that `1` returns
`inf` for a quantity that is mathematically undefined, and that `True` is
silently read as a base. A base is a domain parameter, and the repository's
fail-closed principle (engineering decision principle 9) says an input the
package cannot honour is refused at the earliest knowable stage with a sentence
that names the reason. `0.5` is deliberately **not** in the refused set: it is a
valid base, both packages accept it, and both return the same sign.

The refusal happens in the request's constructor **and** the divisor is computed
again at measurement time, because a user can construct an `OutputRequest`
directly and a plan can be deserialised. `log_base` is stored **unresolved** —
`None` stays `None` rather than being folded to a number — so the request remains
a faithful record of what the caller asked for, and a result's request can be
compared with the call that produced it.

## 6. The zero-eigenvalue floor, stated rather than hidden

`_PROBABILITY_FLOOR = 1e-30` lives in `flagquantum/simulation/entropy.py`. The
sum is computed as

```python
positive = probabilities > _PROBABILITY_FLOOR
safe = torch.where(positive, probabilities, torch.ones_like(probabilities))
terms = torch.where(positive, safe * torch.log(safe), torch.zeros_like(safe))
```

which is a floor and not a clamp: a probability below the floor contributes
exactly zero rather than being nudged to `1e-30`. The recorded pure rows all
contain exact zeros (a Bell pair's reduced state has two), so the floor is
load-bearing on every one of them, and the gate's own reference uses its own
`1e-30` floor so that the comparison is against the same stated convention rather
than against an unstated one.

The gate does **not** currently distinguish `>` from `>=` at the floor, because
no recorded spectrum carries an eigenvalue exactly equal to `1e-30`. That is a
stated blind spot rather than a claimed coverage, and it is recorded in
[the change record](../api-changes/FQ-VN-ENTROPY-OUTPUT-20261101.md).

## 7. The numerics boundary, stated honestly

The contract's `mode_tolerance` and `independent_reference_tolerance` are both
`1e-12`, and that number is defensible only because **every recorded baseline is
`complex128`**:

| Reading | Largest measured difference |
| --- | --- |
| gate reference vs. the package, `complex128` | `7.7e-17` |
| across the five execution modes, `complex128` | `5.4e-15` |
| the same on a freshly rebuilt table | `1.915e-15` |
| the two mixed rows vs. the gate's reference | `0` (exactly) |
| `complex64` against `complex128` on the same program | `2.184e-06` |
| the MPS Schmidt shortcut against the dense matrix, `n=5`, cut 4 | `2.325e-06` |
| a batched MPS run against the dense matrix | `1.431e-06` |

`eps(complex64) = 1.1920928955078125e-07`, so the single-precision story is a
three-order-of-magnitude looser one and the contract does not claim it. A user
who needs the tight tolerance reads it off a `complex128` circuit, exactly as the
recorded rows do.

## 8. What the change does not do

- **No `fq.variance`, no `fq.mutual_info`, no `fq.purity`.** They were authorized
  in the same grant and each lands separately. Nothing in this change reserves
  their names or their signatures.
- **No state/statevector output kind.** `result.measurement(...)` and
  `result.statevector()` are unchanged.
- **No sampling.** Entropy is not a sampled quantity; `shots=` is refused with
  `shots requires fq.samples(...) or fq.counts(...) output` rather than being
  approximated by a histogram. A `stabilizer`-mode run is refused by name for
  the same reason.
- **No new `ExecutionOptions` mode.** The five existing modes are dispatched to;
  none is added.
- **No readout handling.** A noise model carrying a readout rule is refused,
  because readout confusion acts on outcomes after the state and an entropy
  carrying it would describe a state the engine never prepared.
- **No change to `IR_VERSION`,** to any serialized schema, or to any existing
  signature or default.
- **No dependency.** The arithmetic uses `torch` only, which is already the one
  core dependency.

## 9. Why a root export rather than a namespace

The four siblings — `counts`, `expectation`, `probabilities`, `samples`,
`density_matrix` — are all root exports, and `fq.run(..., outputs=...)` is
documented as the single route by which an execution is asked for anything. A
name reachable only as `fq.experimental.measurement.vn_entropy` would be the one
output kind a user could not find where they found the others, and the request
type it returns is `OutputRequest`, which is Stable Core. The export count moves
37 → 38, which is inside the authorized headroom of the measurement-parity grant.

The new name deliberately does not accept the deprecated `wires=` spelling its
five siblings carry:

```python
fq.vn_entropy(wires=[0])   # TypeError: unexpected keyword argument 'wires'
fq.vn_entropy([0])         # the supported spelling
```

`counts`, `expectation`, `probabilities`, `samples` and `density_matrix` each
carry a `wires=` alias as recorded 0.3.x migration debt with a removal version.
A name born after the migration started would be born deprecated, so this one is
not given the alias. This is the same decision Proposal 068 recorded for
`fq.density_matrix`, and it is recorded here again because it is the same
decision.

## 10. Compatibility

| Surface | Before | After |
| --- | --- | --- |
| `fq.__all__` | 37 names | 38 names |
| `fq.vn_entropy` | absent | `(qubits, *, log_base=None, name=None) -> OutputRequest` |
| `fq.OutputRequest` kinds | `counts`, `density_matrix`, `expectation`, `probabilities`, `samples` | plus `vn_entropy` |
| `OutputRequest` fields | `kind`, `qubits`, `observable`, `name` | plus `log_base=None` |
| `ExecutionResult` properties | `.probabilities`, `.counts`, `.density_matrix`, `.expectations` | plus `.vn_entropy` |
| `IR_VERSION` | `"1.0"` | `"1.0"` |
| serialized schemas | unchanged | unchanged |

`OutputRequest.log_base` defaults to `None` and is refused on every other kind,
so the new field is additive for existing callers and cannot be set on a request
that has no meaning for it.

An existing caller who never asks for an entropy sees no change at all.

## 11. Open questions

1. **The floor's exact boundary is unmeasured.** No recorded spectrum carries an
   eigenvalue exactly at `1e-30`, so the gate does not distinguish `>` from
   `>=`. Closing it needs a program whose reduced spectrum is exactly
   representable at that value, which may not exist in `complex128`; the honest
   answer may be to record the blind spot permanently rather than to manufacture
   a row for it.
2. **`state_separation_floor = 1e-6` is a judgement.** It exists so that "which
   side was asked about" is a measurable question rather than a tautology. A
   smaller floor would still catch the failure mode; the recorded value was
   chosen above the measured noise on the mixed rows (`2.2e-16`) by a wide
   margin deliberately, so the row does not become precision-sensitive.
3. **A single-tolerance story is claimed for `complex128` only.** Whether a
   `complex64` row should be recorded with its own looser tolerance, or whether
   the single-precision gap (`2.184e-06`) should be stated in the user-facing
   documentation as well, is unresolved.

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

The two test files are deliberately different jobs. The contract file holds the
gate's tamper battery plus one shortest-path scenario. The scenario file holds the
workflows the recorded numbers describe — a base of two, the same number from four
modes, the selector's side on a mixed state, the 11-qubit program the dense route
refuses by capacity, and a descending or interior selection — each checked against
a reduction and diagonalisation written independently in the test. A contract test
that only re-runs the gate would leave those workflows with no readable home, which
is what guardrail 6 asks for.

The gate's mutation battery is the evidence that its measurements can fail.
Eleven defects were injected one at a time — ten into the package source and one
into the recorded contract — and each was reported:

| Injected defect | Reported |
| --- | --- |
| The natural logarithm is reported for every base | `log base 'base two' answered 0.6931471805599454 rather than the recorded 1.0000000000000002` |
| The selection is read as its complement | `baseline 'damped superposition, the damped qubit' on [0] answered 2.2204460492503126e-16 rather than the recorded 0.19646697846469818` |
| A zero eigenvalue reaches `log(0)` | `baseline 'bell pair, both qubits' on [0, 1] is nan, which no tolerance can compare` |
| The MPS shortcut is forced onto every selection | three rows reported with the wrong `mps` reading |
| The MPS reduced matrix traces the wrong sites | three rows reported with the wrong `mps` reading |
| The MPS prefix spectrum is scaled by a half | `answers 'mps' with … and the default route with …` |
| The entropy's sign is flipped | `baseline 'bell pair, one qubit' on [0] answered 0.0 rather than the recorded 0.6931471805599454` |
| A recorded value is perturbed | `rather than the recorded` |
| The selection's base is ignored inside the entropy arithmetic | `log base 'base ten' answered 0.6931471805599454 rather than the recorded 0.3010299956639812` |
| A refusal sentence is reworded | `refusal 'no qubit is named' does not say 'vn_entropy output requires at least one qubit'` |
| Any base is accepted | `refusal 'a base of zero is refused rather than answered by accident' did not raise` |

The `nan` row is why the gate checks every reading for finiteness **before** it
compares it: `nan > tolerance` is false, so a `nan` answer would have satisfied
every tolerance test and been reported as agreement. That defect was injected
first against a gate without the finiteness check, and the gate stayed green; the
check was added because the mutation was not caught, which is the only honest
reason to add one.

**One blind spot remains, and it was measured rather than assumed.** The gate
does not distinguish `>` from `>=` against `_PROBABILITY_FLOOR`: replacing the
comparison reported nothing and the gate passed, because no recorded spectrum
carries an eigenvalue exactly equal to `1e-30`, so the two comparisons are
indistinguishable on the whole recorded row set. That is a stated blind spot and
not a claimed coverage; § 6 records the floor itself.

An earlier candidate blind spot was checked in the same battery and turned out
**not** to exist: forcing `log_base_divisor(None)` inside
`entropy_from_probabilities` — ignoring the caller's base inside the arithmetic
while the constructor still validates it — is caught
(`log base 'base ten' answered 0.6931471805599454 rather than the recorded
0.3010299956639812`), because the base rows compare the package against the
gate's own arithmetic rather than only against a validated request.
