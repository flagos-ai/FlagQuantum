# Remote

`flagquantum.remote` contains adapters for compute reached through an external
task control plane. A remote target may be a QPU, GPU cluster, classical HPC
service, or cloud platform. The defining property is the submit/status/result
lifecycle—not whether the service is called a quantum cloud.

Each adapter discovers target capabilities, validates a submission, invokes
the external API or SDK, and converts returned data into FlagQuantum-owned
results. Directly controlled devices belong in `flagquantum.compute`.

It does not own deployment-package semantics, Runtime scheduling, compiler
lowering, or simulation numerics. The maintained external adapters are
`braket.py` and `quafu.py`; `http.py` provides their reusable HTTP transport
boundary and `result_parsing.py` normalizes remote counts. Quafu calibration
conversion lives beside its adapter in `quafu_calibration.py`. The in-memory
control-plane test double lives in `flagquantum.testing`; it is not a local
compute implementation. Run `python -m pytest
tests/test_amazon_braket_provider.py tests/test_cloud_providers.py
tests/test_quafu_calibration.py -q`.
