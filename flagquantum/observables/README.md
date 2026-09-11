# Observables

This package owns the public mathematical observable model and the output
requests accepted by `fq.run` and `fq.plan`. It lowers those values to Core IR
measurement nodes; it does not execute numerical kernels, choose resources, or
adapt providers.

Start with `fq.expectation(fq.X(0) @ fq.X(1))` or
`fq.probabilities(wires=(0, 1))`. Run the focused tests with:

```bash
pytest tests/api_contract/test_observable_outputs.py
```

Add a public output kind only when the Runtime can implement the same semantics
across its claimed backends and can fail closed elsewhere.
