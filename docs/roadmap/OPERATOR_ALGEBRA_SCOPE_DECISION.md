# A matrix is a value, not an operator-algebra type

## The decision

**A matrix-valued operator type is not added to the operator algebra.** A matrix stays a
value that a program, a channel or a map consumes, and it does not become an algebra
member that composes with `Observable` or with the fermionic sum through products, sums,
scalars, adjoints or commutators.

This is an owner decision, taken on 2026-10-22, and it is a scope decision rather than a
statement about effort: the work is not queued, and no future revision of
`contracts/cudaq-parity-matrix.toml` should record it as a gap to be closed. The two rows
the decision governs are `matrix_and_custom_operator` in
`operator_algebra_and_domain_libraries` and `kraus_channel_algebra` in
`noise_and_error_models`. They move in **opposite** directions, because the decision
answers the question each one was waiting on, and the answer is not the same answer for
both: one row was waiting to *gain* a type, and the other was waiting for a type it does
not need.

## What a user can do, measured on this tree

Four routes already carry a matrix into a FlagQuantum computation. Every number and every
string below is an output of the code beside it, run in this checkout.

```python
import torch
import flagquantum as fq
from flagquantum.operators import SuperOperator
from flagquantum.noise.channels import amplitude_damping_channel

pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
identity = torch.eye(2, dtype=torch.complex128)

# 1. A matrix is program content.
print([(i.name, i.wires) for i in fq.Circuit(1).any(0, matrix=pauli_x).to_ir().instructions])
# [('any', (0,))]

# 2. A matrix is a factor of a superoperator.
print(SuperOperator.left_right_multiply(pauli_x, identity).dense().shape)
# torch.Size([4, 4])

# 3. A matrix is a Kraus operator.
channel = amplitude_damping_channel(0.3)
print(len(channel.kraus), type(channel.kraus[0]).__name__)
# 2 Tensor

# 4. A matrix acts on a dense state.
from flagquantum.simulation.density_matrix import apply_unitary_density, expand_operator
print(callable(apply_unitary_density), callable(expand_operator))
# True True
```

The matrices on all four routes are ordinary `torch.Tensor` objects. That is the reason
the fifth route is not built. Route 2 also shows the boundary the algebra does keep: a
factor must carry a complex dtype, and a real `torch.eye(2)` is refused as
`an operator factor must have a complex dtype, got torch.float32`, which is a statement
about the operand's dtype rather than about its type.

## What the decision declines, and how it shows up

What a caller cannot write is an arithmetic expression that mixes an operator with a
matrix. Today each of those is a `TypeError` rather than a silent coercion, which is the
behaviour the decision keeps:

```python
import torch
import flagquantum as fq

pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
for label, call in (
    ("X(0) + matrix", lambda: fq.X(0) + pauli_x),
    ("X(0) @ matrix", lambda: fq.X(0) @ pauli_x),
    ("matrix + X(0)", lambda: pauli_x + fq.X(0)),
):
    try:
        call()
    except TypeError as error:
        print(label, "->", error)
# X(0) + matrix -> unsupported operand type(s) for +: 'Observable' and 'Tensor'
# X(0) @ matrix -> unsupported operand type(s) for @: 'Observable' and 'Tensor'
# matrix + X(0) -> unsupported operand type(s) for +: 'Tensor' and 'Observable'
```

The same expression with two observables works, so the refusal is about the type of the
operand and not about the operator:

```python
print(type(fq.X(0) @ fq.Y(1)).__name__)
# Observable
```

## Why the type is not added

Four reasons, each checkable rather than asserted.

**1. The algebra already exists, and it is `torch`.** A matrix in this repository is a
`torch.Tensor`, so it already has products, sums, scalar multiples, an adjoint (`mH`) and
a dtype- and device-aware transpose. A `MatrixOperator` type would be a second algebra
over the same objects, with its own rules for each of those six operations and its own
place for them to disagree with `torch`. The completeness relation of a Kraus set —
`sum_k K_k^dagger K_k = I` — is arithmetic a caller can already write:

```python
import torch

k1 = torch.tensor([[1.0, 0.0], [0.0, 0.8366600265340756]], dtype=torch.complex128)
k2 = torch.tensor([[0.0, 0.5477225575051661], [0.0, 0.0]], dtype=torch.complex128)
print([complex(round(float(v.real), 6)) for v in (k1.mH @ k1 + k2.mH @ k2).diagonal()])
# [(1+0j), (1+0j)]
```

**2. The superoperator algebra already admits a matrix-operator type, and it did so before
this decision.** `SuperOperator` takes terms of `coefficient * left @ state @ right`, and
it dispatches a factor on one `isinstance` arm only: a `torch.Tensor`. Everything else is
duck-typed on `dimension`, `__matmul__`, `__rmatmul__` and `dense()`, with `adjoint()`
required only when the term is actually adjointed. A class wrapping a matrix therefore
joins that algebra today, with no contract edit:

```python
import torch
from flagquantum.operators import SuperOperator


class Wrapped:
    """A matrix operator that is neither a torch.Tensor nor a PauliSum."""

    def __init__(self, matrix):
        self._matrix = matrix

    dtype = property(lambda self: self._matrix.dtype)
    dimension = property(lambda self: int(self._matrix.shape[0]))
    def __matmul__(self, state): return self._matrix @ state
    def __rmatmul__(self, state): return state @ self._matrix
    def dense(self): return self._matrix
    def adjoint(self): return Wrapped(self._matrix.mH)


pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
term = SuperOperator.left_right_multiply(
    Wrapped(pauli_x), torch.eye(2, dtype=torch.complex128)
)
print(term.dimension, tuple(term.dense().shape))
# 2 (4, 4)
print(tuple(term.adjoint().dense().shape))
# (4, 4)
```

So what the decision declines is narrower than "a matrix cannot be an operator": it
declines membership in the `Observable` algebra, the one whose products and sums a
measurement route reads. The factor contract was built as an `isinstance` arm plus duck
typing precisely so that a matrix-free `PauliSum` can join a term without being
materialized; teaching `Observable` a fourth operand shape would put a type hierarchy
where the design chose two implementations.

**3. A matrix type would not make a matrix executable where it is not.** The reason a
declared custom operation is unusable without a per-call matrix is that a declaration
binds no operand, which is a different absence and stays open in
`custom_operation_registration`. Joining the algebra would give a matrix sums and
products; it would not give an instruction its operand.

**4. The absence is a convenience, not a capability, because the algebra can read a
matrix in.** A matrix is recovered as an algebra member through the normalized Pauli
basis without a new type, exactly and in four `trace` calls at one qubit:

```python
import torch
from flagquantum.simulation.pauli import normalized_pauli_basis, pauli_words

basis = normalized_pauli_basis(1, dtype=torch.complex128, device=torch.device("cpu"))
hadamard = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.complex128) / 2 ** 0.5
coefficients = [torch.trace(b.mH @ hadamard) for b in basis]
print(pauli_words(1))
# ('I', 'X', 'Y', 'Z')
print([complex(round(float(c.real), 6)) for c in coefficients])
# [0j, (1+0j), 0j, (1+0j)]
print(bool(torch.allclose(sum(c * b for c, b in zip(coefficients, basis)), hadamard, atol=1e-12)))
# True
```

So the projection route is `H = (X + Z) / sqrt(2)` read off the basis and rebuilt to
`1e-12`. A caller who needs a Pauli sum from a matrix writes those four lines; a caller
who needs to apply a matrix to a state has routes 1 and 4 above. What is missing is
spelling, and the decision is that the spelling is not worth a second algebra.

## What this does not decide

The decision answers one half of the operator-value question and leaves the other half
open. **Whether a registered declaration should bind its operand** — a matrix or a
decomposition — so that the name alone is executable is a separate question, and
`custom_operation_registration` still records it as that row's remaining blocker. Nothing
here says a declaration must keep requiring a per-call matrix; it says only that the way
to close that gap is not a new operator-algebra type.

`kraus_channel_algebra`'s own surface is unaffected and stays where it was built: a
validated Kraus constructor with the completeness relation compared at `atol = rtol =
1e-6`, a versioned `flagquantum.kraus_channel.v1` payload, the factories, the
unitary-mixture classification, and the `NoiseModel` and `noisy_density_matrix` routes.
The row named the missing type as its *only* remaining absence, so the decision removes
its reason to stay `partial` rather than removing work from it.

## Bookkeeping

| Row | Was | Is | Why the direction |
| --- | --- | --- | --- |
| `matrix_and_custom_operator` | `partial`, `next` | `out_of_scope`, `excluded` | Its residual was the declined type. The `custom_op` half of the same CUDA-Q feature is tracked by `custom_operation_registration`, which the decision does not touch. |
| `kraus_channel_algebra` | `partial`, `now` | `supported`, `now` | The declined type was the only absence it recorded. |

`out_of_scope` is used here for the first time in this matrix. It is the right token and
not a way out of a hard row, because the two rows differ in kind: a row that is
`unsupported` is capability this programme still owes, and a row that is `out_of_scope` is
capability the programme has decided not to owe. The generator already treats the second
as outside the open-gap count — `tools/parity_matrix.py` counts `unsupported` and
`partial` only — so the scoreboard reports the smaller target rather than an unfinished
one. That is the honest report, but it is also a target that shrank, and § 3 of
[`CUDAQ_PARITY_STRATEGY.md`](CUDAQ_PARITY_STRATEGY.md) is where the replacement
obligation is stated: this decision changes the parity row set, not the obligation the
`A_nvidia_proprietary` rows carry.

Two machine rules keep the token from becoming the status edit that § 6 forbids. The
validator now requires an `out_of_scope` row to carry `excluded` priority and a
`decision_ref`, to point at a file that exists, and to point at a file that names the row
by id, so a row and its record cannot drift apart in either direction. And
`tests/unit/test_parity_scope_rows.py` holds the record's central claim about the tree: it
fails if a `MatrixOperator`-shaped name appears under `flagquantum/` or in the Stable Core
export set, because the decision would have been reversed by implementation and the record
would be describing a tree that no longer exists.

Moving `kraus_channel_algebra` the other way needed no new evidence, because the row
already cited the `noisy_simulation` registry entry and a `supported` status requires one.
What it did need was for the rest of its surface to be measured instead of inherited from
its own prose, which is why its reason now carries the ten factories, the `<Z> = -0.4`
check, the serialization round trip and the `from_kraus` route as measurements rather than
as claims that a constructor exists.

## What would reopen this decision

A concrete caller whose program needs matrix algebra *and* cannot express it through the
four routes or the Pauli projection, at a cost that matters. The cheapest such caller is
one that needs a product of two matrices that are each already an algebra member, because
that is the one expression the projection route does not shorten. A request of that shape
should reopen the question with the program that needs it, rather than accumulate a type
on the chance one will.
