# Execution providers

`flagquantum.providers.execution` contains adapters for external quantum
execution targets. Each adapter discovers target capabilities, validates a
submission, invokes the provider SDK, and converts provider results into
FlagQuantum-owned results.

It does not own deployment-package semantics, Runtime scheduling, compiler
lowering, or simulation numerics. Provider adapters live in modules such as
`braket.py`, `cqlib.py`, `fieldquantum.py`, `originq.py`, `quafu.py`, and
`tencent.py`; shared HTTP transport and response normalization live in
`http.py` and `result_parsing.py`. Run `python -m pytest
tests/test_amazon_braket_provider.py tests/test_cloud_providers.py -q`.
