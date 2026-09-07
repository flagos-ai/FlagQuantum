# Execution providers

`flagquantum.providers.execution` contains adapters for external quantum
execution targets. Each adapter discovers target capabilities, validates a
submission, invokes the provider SDK, and converts provider results into
FlagQuantum-owned results.

It does not own deployment-package semantics, Runtime scheduling, compiler
lowering, or simulation numerics. Start with `braket.py` for an adapter example;
shared provider-response normalization lives in `result_parsing.py`. Run
`python -m pytest tests/test_amazon_braket_provider.py -q`.
