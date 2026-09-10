# Jiuding direct program submission

Updated: 2026-09-10

Status: **experimental adapter implemented; no Stable Core API change**

## User path

Users can submit a `Circuit` or `CircuitIR` as an isolated Jiuding batch job
without creating a Python `main()` script:

```python
import flagquantum as fq
from flagquantum.remote.compute.jiuding import JiudingClient

client = JiudingClient(workspace="my-workspace")
receipt = client.submit_program(
    fq.Circuit(2).h(0).cx(0, 1),
    target="jiuding:gpu",
    image="flagquantum-dev:v1",
    receipt="/shared/jobs/bell.json",
    outputs=fq.counts(),
    shots=1024,
)
result = client.result(receipt, timeout=600)
print(result.counts)
```

The receipt location remains explicit because the batch container and caller
must see the same durable storage. Image selection also remains explicit; an
unverified image is never guessed. The adapter writes a bounded program request
and minimal runner beside the receipt, submits through the existing idempotent
job path, and restores a normal `ExecutionResult`.

Pass `pythonpath=` only when the image provides the runtime dependencies but
the job should load a newer FlagQuantum source tree from shared storage.

## Boundaries

- Existing script-based `submit()` behavior is unchanged.
- Program execution uses the ordinary Runtime and requested CPU or GPU device.
- CPU fallback is not enabled or hidden.
- Receipt identity remains available after client-process failure.
- No root `fq.submit` API is introduced before a provider-neutral Job contract
  has evidence from more than one external control plane.
