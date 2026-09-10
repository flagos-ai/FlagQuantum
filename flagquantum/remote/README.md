# Remote

`flagquantum.remote` contains adapters for compute reached through an external
task control plane. A remote target may be a QPU, GPU cluster, classical HPC
service, or cloud platform. The defining property is the submit/status/result
lifecycle—not whether the service is called a quantum cloud.

Each adapter discovers target capabilities, validates a submission, invokes
the external API or SDK, and converts returned data into FlagQuantum-owned
results. Directly controlled devices belong in `flagquantum.compute`.

It does not own deployment-package semantics, Runtime scheduling, compiler
lowering, or simulation numerics. Existing quantum task contracts and adapters
live under `qpu/`; their shots-and-counts result model must not become the
contract for future remote compute services. Add `compute/` only when a real
remote GPU or HPC adapter establishes that lifecycle. The in-memory
control-plane test double lives in `flagquantum.testing`; it is not a local
compute implementation. Run `python -m pytest
tests/test_amazon_braket_provider.py tests/test_cloud_providers.py
tests/test_quafu_calibration.py -q`.

Import QPU providers from `flagquantum.remote.qpu` and `JiudingClient` from
`flagquantum.remote.compute`. The categories remain separate at the public
boundary; neither is flattened into another generic remote backend API.

The first classical compute adapter lives under [`compute/`](compute/README.md).
Its experimental Jiuding path submits CPU or GPU work from an existing
workspace and reads a shared result; it does not extend the QPU result model.
