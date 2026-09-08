# Execution providers

`flagquantum.providers.execution` contains adapters for external quantum
execution targets. Each adapter discovers target capabilities, validates a
submission, invokes the provider SDK, and converts provider results into
FlagQuantum-owned results.

It does not own deployment-package semantics, Runtime scheduling, compiler
lowering, or simulation numerics. The maintained adapters are `braket.py`,
`local.py`, and `quafu.py`; `http.py` provides their reusable HTTP transport
boundary and `result_parsing.py` normalizes remote counts. Quafu calibration
conversion lives beside its adapter in `quafu_calibration.py`. Run `python -m pytest
tests/test_amazon_braket_provider.py tests/test_cloud_providers.py
tests/test_quafu_calibration.py -q`.
