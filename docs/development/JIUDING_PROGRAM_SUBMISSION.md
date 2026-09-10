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
    image="flagquantum-runtime:v1",
    outputs=fq.counts(),
    shots=1024,
)
result = client.result(receipt, timeout=600)
print(result.counts)
```

By default, the adapter uses the selected workspace's SSH connection to stage a
bounded request, a minimal runner and a content-addressed FlagQuantum source
snapshot under `/share/project/.flagquantum`. It submits through the existing
job path and fetches the run-bound result through the same connection. Repeated
submissions of unchanged source reuse the snapshot. Image selection remains
explicit; an unverified image is never guessed.

Pass `receipt=` to use expert-managed shared storage instead. In that mode,
`pythonpath=` may select a shared FlagQuantum source tree supplied by the user.

## Boundaries

- Existing script-based `submit()` behavior is unchanged.
- Managed transfer requires a running workspace with SSH and writable
  `/share/project`; failure is explicit and never falls back silently.
- Program execution uses the ordinary Runtime and requested CPU or GPU device.
- CPU fallback is not enabled or hidden.
- Receipt identity remains available after client-process failure.
- No root `fq.submit` API is introduced before a provider-neutral Job contract
  has evidence from more than one external control plane.
