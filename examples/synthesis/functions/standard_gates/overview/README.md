# Native standard-gate tutorial

Run all nine native quantum functions from the reviewed Notebook:

```sh
python -m examples.synthesis.functions.standard_gates.overview.main
```

The [workflow guide](WORKFLOW.md) describes the source-to-native mapping and
verification. Each imported function is a readable native `@qfunc` with direct
calls such as `R(1, 2, q)` and `U(1, 2, 1.5, 1.1, q)`. Follow the corresponding
[R example](../r/README.md) or [U example](../u/README.md) for its function body and
independent Circuit command.
This workflow does not load external quantum source text.

[Third-party license](../../../CLASSIQ_LICENSE.txt)
