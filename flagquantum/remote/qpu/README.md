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

For the shortest supported user path, let the public runtime own the complete
compile, pre-submission validation, submission, wait, and result sequence:

```python
import flagquantum as fq

circuit = fq.Circuit(2).h(0).cx(0, 1)
result = fq.run(
    circuit,
    compiler="qsteed",
    target="quafu:Baihua",
    shots=1024,
)
print(result.provenance["task_id"], result.counts[0])
```

Run [`examples/remote/quafu_bell.py`](../../../examples/remote/quafu_bell.py)
after setting `QUAFU_API_TOKEN`. Use `QuafuProvider` directly only for backend
discovery or an advanced package-oriented workflow.

`QuafuProvider` accepts an already compiled deployment package or precompiled
OpenQASM with an explicit physical-qubit mapping. `AmazonBraketProvider`
adapts an `AwsDevice` and provides a non-submitting `dry_run()` before task
creation. `AzureQuantumProvider` adapts one QDK workspace target, compiles a
sealed OpenQASM 3 deployment package to QIR, and normalizes returned counts or
probabilities. Generic HTTP transport components support concrete provider
adapters; they are not a separate execution product.

The Azure route needs the optional Microsoft QDK client, and nothing else in
FlagQuantum does. It is opt-in, so a default install stays without it:

```bash
python -m pip install -e '.[azure]'   # adds qdk[azure]
```

Only a real run reaches an Azure workspace; the offline provider tests pass a
fake workspace and target and never read a credential. Azure configuration is
required at that point and not before, through environment-owned values:

```bash
export AZURE_QUANTUM_RESOURCE_ID="<workspace-resource-id>"
export AZURE_QUANTUM_TARGET_QUBITS="<authoritative-target-width>"
```

```python
import flagquantum as fq

result = fq.run(
    fq.Circuit(2).h(0).cx(0, 1),
    outputs=fq.counts(),
    target="azure:quantinuum.qpu.h2-1",
    shots=100,
)
```

This route currently supports full-register counts only. It rejects unsupported
outputs, physical-qubit mappings, and missing target-width evidence before
creating a cloud job. Use `AzureQuantumProvider.dry_run()` directly when an
application needs a non-submitting package preview.

Change high-level remote `fq.run` workflows in `execution.py`; change each
provider's transport and task lifecycle behavior in its provider module.

Follow the [Quafu guide](../../../docs/guides/QUAFU_BACKEND.md) for the complete
compile, map, submit and result path. A provider adapter must preserve task and
deployment identity and must not silently retry an ambiguous submission.

Run the offline provider checks from the repository root:

```bash
python -m pytest tests/test_amazon_braket_provider.py \
  tests/test_azure_quantum_provider.py tests/test_cloud_providers.py \
  tests/test_quafu_calibration.py \
  tests/team/remote/test_quafu_user_path.py -q
```
