# Hubbard graph training across three encodings

## Scope before implementation

Start from accepted main `6d9ec4aac2ec8d330969958327dacabfaf65e1d1`.
Extend the existing four-site star example with an explicit `--encoding`
selection for JW, prefix parity and Fenwick BK. Keep JW as the default.
Reuse the existing constructor, encoder, native expectation and PyTorch SGD.
Only the example's fixed determinant-pair preparation changes with the basis;
this is not a general ansatz conversion API.

The physical pair remains cos(theta/2)|00010010> +
sin(theta/2)|10010000>. Its encoded words, qubit zero first, are:

| Encoding | First determinant | Second determinant |
| --- | --- | --- |
| JW | 00010010 | 10010000 |
| Prefix parity | 00011100 | 11100000 |
| Fenwick BK | 00010010 | 11000000 |

Prepare these words with X, RY and CX gates, without dense state loading,
eigensolvers or an external SDK. The spectator sign gives
E(theta)=1.2*sin(theta)-0.4 in every basis. The pair minimum -1.6 is not the
full graph ground energy. Only the circuit angle is trainable.

## Acceptance before implementation

- Check complete prepared amplitudes at several signed angles against the
  independently specified encoded words; a wrong relative phase must fail.
- Replace encoding and its matched fixed preparation without modifying the
  measurement/SGD consumer. Verify analytic energy and gradient, finite
  differences, optimizer updates and one particle of each spin in each basis.
- Replace path/cycle producers at the same consumer in each encoding.
- Execute all three CLIs with external science SDKs and dense eigensolvers
  blocked. Reject unknown encoding and nonscalar preparation angles.
- Preserve the default invocation and existing graph/chain/ring interfaces.

Ownership is Algorithms. This CPU `single_device_fast_path` slice adds no
dependency, Stable Core export, capability registration, solver, automatic
sector selection, tapering, coefficient gradient or scalability claim. Existing
graph registration remains historical; any metadata expansion is a separate
Integration-owned change after this implementation's acceptance.


## Contributor workflow

```bash
python -m examples.algorithms.fermion_hubbard_graph
python -m examples.algorithms.fermion_hubbard_graph --encoding parity
python -m examples.algorithms.fermion_hubbard_graph --encoding bravyi_kitaev
python -m pytest tests/integration/test_fermion_hubbard_graph.py -q
```

Each CLI should start at 0.0339385184 and reach -1.5997840500 after 40 SGD
steps, with alpha/beta populations 1/1 and finite-difference residual below
1e-9. The eight-qubit register is unreduced. To change the pair preparation,
first check all 256 amplitudes and the two occupied amplitudes' signed angle
derivatives, then rerun
the analytic energy/update tests and isolated CLIs. Keep explicit preparation
inside the example; the encoder continues to own the authoritative binary
basis semantics and the runtime owns execution/autograd.

## Local review evidence

Scope commit `f71be4f8f` preceded implementation. The final branch passed 203
related Hubbard/encoding scientific cases, including 26 graph integration
cases, five selected CLI/inventory cases, and 37 capability/documentation cases.
The graph cases cover all three encodings, signed full-state amplitudes,
analytic and finite-difference gradients, optimizer updates, spin populations,
path/cycle producer replacement and three isolated CLI processes.

Strict mypy on the example and graph integration tests, Ruff and Black on all
three changed Python files, the Stable Core API baseline, generated docs
consistency, Algorithms scope, whitespace and a 1,423-file local-link/anchor
audit passed. Invalid runtime inputs are exercised through parametrized tests
without type-error suppressions.

These are focused local review results. Full Linux verification belongs to PR
CI; this branch does not claim merged-main acceptance or capability promotion.
