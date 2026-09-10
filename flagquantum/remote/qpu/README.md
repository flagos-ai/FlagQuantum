# Remote QPU

This package adapts externally controlled quantum processors. It owns backend
discovery, provider credentials, task submission, status, cancellation and
result decoding. Compiler passes, deployment-package construction, simulation
and numerical post-processing remain in their owning domains.

Use the categorized public entry point:

```python
from flagquantum.remote.qpu import QuafuProvider

provider = QuafuProvider()
provider.verify()
backends = provider.discover_backends(n_wires=2)
```

`QuafuProvider` accepts an already compiled deployment package or precompiled
OpenQASM with an explicit physical-qubit mapping. `AmazonBraketProvider`
adapts an `AwsDevice` and provides a non-submitting `dry_run()` before task
creation. Generic HTTP transport components support concrete provider adapters;
they are not a separate execution product.

Follow the [Quafu guide](../../../docs/guides/QUAFU_BACKEND.md) for the complete
compile, map, submit and result path. A provider adapter must preserve task and
deployment identity and must not silently retry an ambiguous submission.

Run the offline provider checks from the repository root:

```bash
python -m pytest tests/test_amazon_braket_provider.py \
  tests/test_cloud_providers.py tests/test_quafu_calibration.py -q
```
