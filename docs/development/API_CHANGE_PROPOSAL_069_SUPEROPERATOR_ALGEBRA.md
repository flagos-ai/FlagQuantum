# API Change Proposal 069: A superoperator algebra in `flagquantum.operators`

## Status

**Approved by the repository owner for implementation as plan row W6-02a.** The
owner fixed the three decisions this proposal depends on: the algebra lives in a
new `flagquantum/operators` package, W6-02 is split into W6-02a (this algebra)
and W6-02b (the `Liouvillian` replacement), and the public surface is the CUDA-Q
constructor set plus `apply`, a byte-ceilinged `dense`, scalar multiplication,
`__add__` and `__eq__` -- and nothing else.

The surface is additive and lands in a namespace that already exists as a
`stable_extensions` entry, so it changes no Stable Core export, no signature, no
default, no result field, and no serialized schema. Section 3 states exactly
which contract file changes and why the change is one file rather than two.

Six premises below were measured on this checkout before the implementation was
written, because engineering decision principle 6 requires inspecting what
already exists and principle 10 forbids claiming a gap that has not been
observed. Each is followed by the literal command and the literal output.
Nothing in "Required evidence" is a result.

## Problem

### 1. `flagquantum/operators` is a five-line re-export shim

```console
$ wc -l flagquantum/operators.py && cat flagquantum/operators.py
       5 flagquantum/operators.py
"""Stable operator schema and discovery interface."""

from .core.operator_schema import GateInfo, gate_info

__all__ = ("GateInfo", "gate_info")
```

It is a `stable_extensions` namespace with exactly two symbols, and a module
cannot coexist with a package of the same name, so the shim becomes the package
and re-exports the same two names. Both of its importers keep working
unmodified: `tests/unit/test_operator_schema.py:6` and
`tests/test_qiskit_interop_conformance.py:9` both write
`import flagquantum.operators as fqo` and use `fqo.gate_info` / `fqo.GateInfo`.

### 2. The Lindblad generator is already a superoperator; only the algebra is missing

`Liouvillian` holds the generator as separate pieces and applies it in density
form. Its dense form is bitwise equal to the linear combination of left and
right multiplication actions that CUDA-Q builds with `SuperOperator`, on a
fixture whose Hamiltonian has a complex off-diagonal entry so that the right
factor's transpose is distinguishable from its conjugate transpose:

```console
$ PYTHONPATH=$PWD python - <<'PY'
import torch
from flagquantum.simulation.lindblad_generator import Liouvillian

def embed(single, wire, n):
    ops = [torch.eye(2, dtype=torch.complex128)] * n
    ops = list(ops); ops[wire] = single
    out = ops[0]
    for o in ops[1:]:
        out = torch.kron(out, o)
    return out

h = torch.tensor([[0.3, 0.2 - 0.4j], [0.2 + 0.4j, -0.7]], dtype=torch.complex128)
H = embed(h, 0, 2)
l = 0.4 * torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.complex128)
c0, c1 = embed(l, 0, 2), embed(l, 1, 2)
gen = Liouvillian(H, [c0, c1], hilbert_dimension=4)
I4 = torch.eye(4, dtype=torch.complex128)
terms = -1j * torch.kron(H, I4) + 1j * torch.kron(I4, H.T.contiguous())
for c in (c0, c1):
    p = c.conj().T @ c
    terms = terms + torch.kron(c, c.conj()) - 0.5 * (
        torch.kron(p, I4) + torch.kron(I4, p.T.contiguous())
    )
print("dense max abs diff vs kron sum:", float((gen.dense() - terms).abs().max()))
print("bitwise equal                 :", torch.equal(gen.dense(), terms))
g = torch.Generator().manual_seed(5)
raw = torch.randn(4, 4, dtype=torch.complex128, generator=g)
rho = raw @ raw.conj().T
rho = rho / rho.trace()
manual = -1j * (H @ rho - rho @ H)
for c in (c0, c1):
    p = c.conj().T @ c
    manual = manual + c @ rho @ c.conj().T - 0.5 * (p @ rho + rho @ p)
print("derivative max abs diff       :", float((gen.derivative(rho) - manual).abs().max()))
print("trace preservation ||L^T vec(I)||:", float(torch.linalg.norm(gen.dense().T @ I4.reshape(-1))))
PY
dense max abs diff vs kron sum: 0.0
bitwise equal                 : True
derivative max abs diff       : 0.0
trace preservation ||L^T vec(I)||: 0.0
```

So the mathematics is present and verified; what is absent is a *type*. CUDA-Q
exposes one (`cudaq.SuperOperator`), the parity contract records the row as
`partial` for exactly this reason, and a user of FlagQuantum cannot write the
Lindblad generator as a sum of left and right multiplication actions.

### 3. The right factor of the dense form is transposed, never conjugate-transposed

`flagquantum/simulation/lindblad_generator.py:126-129` records the convention:
`vec(A rho B) = (A (x) B^T) vec(rho)` for the row-major `Tensor.reshape(-1)`.
`Liouvillian.dense()` therefore uses `torch.kron(operator, torch.conj(operator))`
== `torch.kron(L, (L^dag)^T)` because the caller passes `L^dag` as the right
factor. Any new dense form must use the same rule, or the two representations
disagree on the first non-Hermitian right factor.

### 4. `KrausChannel` has no composition, so a channel product is a genuine gap

```console
$ PYTHONPATH=$PWD python -c "
import torch, flagquantum.noise.channels as c
from flagquantum.noise.channels import depolarizing_channel
ch = depolarizing_channel(0.1)
print('matmul  :', hasattr(ch, '__matmul__'))
print('mul     :', hasattr(ch, '__mul__'))
print('compose :', hasattr(ch, 'compose'))
"
matmul  : False
mul     : False
compose : False
```

The composition of two channels is `T @ S` in the vectorized representation, so
an algebra with `__matmul__` would be the substrate for it. This proposal does
**not** add it: the owner excluded the Choi/PTM/Kraus representation transforms
from W6-02a, and a channel-composition API needs its own evidence that the Kraus
and superoperator forms agree.

### 5. There is no matrix-free operator carrier in Core

`flagquantum/core/` has no matrix-valued operator type: `OPERATOR_SCHEMAS`
describes circuit opcodes, and the only matrix-free carrier in the repository is
`flagquantum/simulation/matrix_free_hamiltonian.py`'s `PauliSum`, which Core may
not import (`architecture.toml` `[boundaries] core_forbidden` contains
`"simulation"`).

That constraint fixes the factor type of this algebra. A superoperator factor is
whatever the algebra *does to a matrix*: it left- or right-multiplies one. The
algebra therefore requires four members of a factor -- `dtype`, `dimension`,
`__matmul__` with a square matrix, and `dense()` -- and `torch.Tensor` satisfies
all four after a two-line normalization. A matrix-free `PauliSum` is the second
concrete use engineering decision principle 11 asks for: it already has `dtype`,
`dimension` and `dense()`, and W6-02b adds the missing `__matmul__`. Restricting
the factor type to `torch.Tensor` would force W6-02b to densify the Hamiltonian
and would turn a matrix-free generator into a `4**n`-entry one.

### 6. CUDA-Q's Python surface is six callables with no matrix and no equality

Measured on `repos/cq` at HEAD `b992378`: `python/cudaq/operators/super_op/__init__.py`
is a single re-export line, `python/runtime/cudaq/operators/py_super_op.cpp`
binds only `SuperOperator()`, the three static constructors as `product_op` and
`sum_op` overloads, `__iter__`, and `__iadd__`, and `num_terms()`, `operator[]`,
`begin()/end()` and `operator*=` exist in C++ but are not bound. `+=` is pure
term concatenation (`runtime/cudaq/operators/super_op.cpp:12-15`), and a term is
an ordered `(left, right)` pair in which a missing side means the other
multiplication.

This proposal keeps the constructor set and the `(left, right)` term shape, adds
the four members W6-02 needs to be usable and testable, and leaves out every
representation transform CUDA-Q also lacks.

## Decision

### 1. `flagquantum/operators` becomes a package

`flagquantum/operators.py` is deleted and replaced by
`flagquantum/operators/__init__.py`, which re-exports `GateInfo` and `gate_info`
from `flagquantum.core.operator_schema` and exports `SuperOperator` from
`flagquantum.operators.superoperator`. The package adds no import-time cost
beyond the schema module it already loaded, and it does not import
`flagquantum.simulation`. A `README.md` in the package states what it owns, what
it must not own, its allowed dependencies, its entry points, and the shortest
path for a typical change, per maintainability guardrail 7.

### 2. The public surface

```text
SuperOperator()                                  no-action linear map
SuperOperator.left_multiply(operator)            A rho
SuperOperator.right_multiply(operator)           rho B
SuperOperator.left_right_multiply(left, right)   A rho B
SuperOperator.dimension                          the side d of the operators
SuperOperator.dtype                              the dtype every view uses
SuperOperator.apply(state)                       the map applied to a (d, d) or (b, d, d) state
SuperOperator.dense(*, max_bytes=...)            the (d*d) x (d*d) row-major matrix
SuperOperator.__iter__                           yields (left, right) pairs, None for a missing side
SuperOperator.__iadd__ / __add__                 concatenation, as CUDA-Q's += is
SuperOperator.__mul__ / __rmul__                 scalar multiplication
SuperOperator.__eq__                             exact structural equality
```

There is no `__matmul__` and no coefficient accessor. Scalar multiplication
scales the term coefficient rather than mutating the factors, because CUDA-Q's
`operator*=` folds the scalar into the operators and a matrix-free factor has no
in-place scale here. The observable behaviour is identical: `c * map` multiplies
both present factors by `c`.

### 3. One contract file changes

`contracts/public-api-v1-candidate.json` gains `SuperOperator` in the
`flagquantum.operators` `stable_extensions` entry's `symbols` list **and** in
`approved_namespace_additions`. Both edits are required, and neither alone is
sufficient: `test_candidate_classifies_every_historical_stable_export_exactly_once`
compares the classified symbol set against the baseline plus the authorized
additions, so a declared symbol without the matching authorization fails, and
`test_every_stable_migration_destination_is_importable` imports every declared
symbol, so an authorization without the declaration fails. API Change Proposal
067 measured both negative controls on this same file by patching it and
restoring it; this proposal reuses that result rather than repeating it.

`flagquantum/operators` is a `stable_extensions` entry and not a Stable Core
export, so `docs/public_api_v1.json`, `flagquantum.__all__` and
`contracts/public-api-v0.2-baseline.json` are untouched.
`test_migrated_exports_are_not_discoverable_at_root` keeps `SuperOperator` out
of `dir(flagquantum)`.

### 4. Two other integration surfaces change

`architecture.toml` gains `"operators"` in
`[package_layout] allowed_top_level_directories`; the layout check enumerates the
directories that actually contain Python files and rejects one that is not
listed.

`team-ownership.toml` replaces the dead-ended `flagquantum/operators.py` entry in
`[teams.core].owns` with `flagquantum/operators/**`. Keeping both would leave the
first entry matching no path, which `dead_owns_entries` reports and which
`tests/unit/test_team_scope_policy.py` compares against a reviewed set.

### 5. `apply` is the single definition of the map; `dense` is the second view

`apply(state)` computes `sum(coefficient * left @ state @ right)` with `@`
handling the batch axis natively. `dense()` sums
`coefficient * torch.kron(left, right.transpose(0, 1).contiguous())` for
right-containing terms, `torch.kron(left, I)` for a left-only term and
`torch.kron(I, right.transpose(0, 1).contiguous())` for a right-only term. The
two are independent formulas, so a test can compare them against each other
rather than against one derivation, and the row-major identity in section 3
above is what makes them agree. `dense()` refuses above `max_bytes` instead of
allocating, with the same entry-count arithmetic `Liouvillian.dense` uses, so
W6-02b can delegate without weakening the ceiling.

### 6. What this proposal does not do

- It does not touch `Liouvillian`. That is W6-02b, whose acceptance item is that
  the 30-odd existing tests in
  `tests/unit/test_lindblad_vectorized_generator.py` pass with `Liouvillian`'s
  public methods unchanged -- the replacement test engineering decision
  principle 10 requires.
- It does not add `PauliSum.__matmul__`. That is W6-02b's other half, and it is
  also a Simulation change.
- It does not add Choi, PTM or Kraus representations, partial trace, `dagger`,
  `trace`, or `to_matrix` on the map. CUDA-Q does not have them either, and
  section 4 of "Problem" records the separate gap they would close.
- It does not promote anything to the root API and does not add a
  `capability-maturity.toml` entry. The `super_operator_algebra` parity row may
  gain `partial`-preserving evidence; promoting it to `supported` requires a
  maturity entry this proposal does not have.

## Required evidence

1. `flagquantum/operators/__init__.py` re-exports `GateInfo`, `gate_info` and
   `SuperOperator`, and both existing importers of the shim pass unchanged.
2. A dense form that is bitwise equal to the explicit Kronecker sum on the
   section-2 fixture, and an `apply` that is bitwise equal to the density-form
   term sum on the same fixture, with a deliberate wrong rule shown to miss the
   tolerance.
3. The row-major identity `vec(A rho B) == (A (x) B^T) vec(rho)` checked against
   `torch.kron` on a complex non-diagonal state, plus the reduced
   `K (x) K.conj()` collapse form.
4. Trace preservation `L^T vec(I) == 0` and the antiunitary-involution
   Hermiticity condition, both computed from `dense()` and from `apply()`.
5. `apply` and `dense` cross-checked against each other on random states, so
   neither is trusted as the definition of the other.
6. The byte ceiling refusing at the documented boundary and admitting at it,
   with the error message naming the size and the alternative.
7. A mutation run over the new module, with the exit-2 weak-kill distinction
   reported separately, and a list of deliberate non-mutations with structural
   reasons.
8. `python tools/ci_tier.py pr-default` and `python tools/ci_tier.py pr-runtime`
   on the final tree, plus the full static guardrail roster.
