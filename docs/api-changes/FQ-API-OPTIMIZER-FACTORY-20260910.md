# Public optimizer factory contract

Status: approved and implemented on 2026-09-10.

## Decision

The user explicitly authorized renaming `optimizer_cls` to
`optimizer_factory`, publishing the `OptimizerFactory` type protocol, and
removing the old parameter without a compatibility layer.

`run_vqe` and `run_adapt_vqe` now expose the following keyword:

```python
optimizer_factory: OptimizerFactory = torch.optim.Adam
```

`OptimizerFactory` is publicly available from `flagquantum.algorithms`. It is a
structural typing protocol for a callable that accepts the trainable tensors and
the keyword-only learning rate, then returns a `torch.optim.Optimizer`.

## Compatibility

The former `optimizer_cls` keyword is removed without an alias or compatibility
layer. Callers must use `optimizer_factory`. This name reflects the actual
contract: classes such as Adam and SGD work, as do compatible factory functions.
Algorithm behavior, defaults, optimizer ownership, and result types are
unchanged.

## Verification

Tests cover Adam, SGD, a custom factory, both VQE entry points, and rejection of
the removed keyword. Formatting, lint, architecture, dependency, and public API
checks are run before the change is committed.
