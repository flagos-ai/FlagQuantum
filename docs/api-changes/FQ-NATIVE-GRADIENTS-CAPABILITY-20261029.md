# FlagQuantum's native gradient capability has one registration, and it disagrees with itself

- **Status:** implemented
- **Owner:** core (`flagquantum/gradients.py` as the surface, with the executor and benchmark
  summaries it is compared against)
- **Date:** 2026-10-29
- **Related:** [`fq.gradient` and the reported gradient method](FQ-GRADIENT-API-20261002.md),
  [Every differentiable opcode in every execution mode](FQ-GRADIENT-OPCODE-EXACTNESS-20261028.md).
  Slice `N3-9`'s gradient method matrix (`FQ-GRADIENT-METHODS-MATRIX-20261027.md`) is the
  other record that measures this surface; it is still under review in its own change and
  is deliberately not linked from here until it lands.

## Problem

FlagQuantum's gradient capability was described in three places, and none of them could
see the other two.

The split real/imag statevector executor summaries report `gradient_method =
"parameter_shift"`. The P5 autograd bridge summary reports `backward_method =
"parameter_shift"`. The reversible adjoint behind the distributed statevector reverse
executor reports `gradient_method = "statevector_adjoint"`. A fourth site, the MPS
tensor-factorization split, reports `gradient_method` as well -- but with the value
`projected_stop_subspace` or `none`, which is the backward rule of a rank truncation and
has nothing to do with differentiating a parameter.

Reading all of them together is what makes the defect visible, and it is not the one the
plan predicted for this slice.

The identifier `gradient_method` carries **two unrelated concepts**: the differentiation
route, and the backward rule of a tensor-factorization split. Within the differentiation
route the vocabularies disagree three ways:

- `parameter_shift` is the only name the executors, the benchmark table and the public
  API all agree on.
- `statevector_adjoint` is declared by the reverse executor, by 20 rows of the benchmark
  engine table, and twice by the hybrid compilation contract -- and it is not a public
  method name. Asking for it is refused as `unknown gradient method`, so the message a
  caller receives says the method does not exist while the capability is implemented and
  is reachable as the backward pass behind PyTorch autograd. Asking for `adjoint` instead
  gets the honest refusal.
- the route the public API calls `autograd` is declared twice more under two other names,
  and one of those names is a sentence: `backpropagation through exact statevector`.

Two of the five public method names are declared by no capability block at all. So the
asymmetry runs in both directions -- a declared identifier no caller can request, and a
served public method nothing declares -- and neither direction was checked by anything.

## What changed

Nothing in the implementation and nothing in the public surface. This slice registers
what is already declared, in `contracts/native-gradients-contract.toml`, and adds a gate
that re-derives each declaration instead of trusting it.

`[[alias]]` is the single registration of the correspondence between a declared value and
the public method it denotes. It is what makes the gate fail closed in both directions:
every differentiation-route value appearing in the package census or in the benchmark
engine table must have an alias row, and the gate drives `fq.gradient` for every public
method name. A new declaration -- in an executor, in the benchmark table, or in the hybrid
contract -- turns the gate red until it is registered here.

An empty `public_method` on an alias row is the record that the declared value is not a
method any caller can request. That is how `statevector_adjoint`,
`projected_stop_subspace` and `none` are recorded.

## Measured state

All figures below are re-derived by `tools/check_native_gradients_contract.py` on every
run, so none of them is a snapshot that can go stale silently.

| Fact | Measured |
| --- | --- |
| `gradient_method` / `backward_method` declaration sites in `flagquantum/**` | 12 |
| Files those sites live in | 10 |
| Axes the identifier is used on | 3 (`differentiation_route` 7, `svd_truncation_rule` 3, `field_passthrough` 2) |
| Benchmark engine rows declaring a gradient method | 22 |
| Engine rows declaring `statevector_adjoint` | 20 |
| Engine rows whose value is prose rather than an identifier | 1 |
| Hybrid compilation phases declaring a gradient method | 2 |
| Public gradient method names the implementation accepts | 5 |
| Public method names declared by any capability block | 1 (`parameter_shift`) |
| Public method names declared by none | 4 (`auto`, `autograd`, `finite_difference`, `spsa`) |
| Declared differentiation-route values no caller can request | 1 (`statevector_adjoint`) |

## Refusals, driven rather than quoted

| Asked for | Raised | Meaning |
| --- | --- | --- |
| `adjoint` | `CapabilityError` | honest: there is no standalone adjoint gradient, and no result reports whether backward used adjoint replay |
| `statevector_adjoint` | `ValidationError` | `unknown gradient method`, although 20 engine rows and 2 hybrid phases declare it |
| `torch_reverse_mode_autograd` | `ValidationError` | the same shape, against the name the PennyLane engine row uses |

## What this slice does not decide

The vocabulary has to converge on one name, and which name wins is a public-surface
decision, not a registration one. `statevector_adjoint` becoming a requestable method
name would be a Stable Core API change and needs its own authorization and proposal; the
plan already carries that question. This slice makes the disagreement impossible to lose
sight of, and leaves the choice where it belongs.

## Verification

```bash
python tools/check_native_gradients_contract.py
python -m pytest tests/unit/test_native_gradients_contract.py -q
```

The gate is wired into `.github/workflows/ci.yml` and `tools/pre_push.py`, and
`tests/unit/test_native_gradients_contract.py::test_the_gate_runs_in_ci_and_before_push`
asserts both, because a gate no workflow invokes is a script.
