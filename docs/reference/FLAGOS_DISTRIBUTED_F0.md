# FlagOS distributed runtime F0

F0 connects FlagQuantum's existing `torch.distributed` runtime to the public
Torch-FL process-group boundary. It accepts `device="flagos"` (or
`device="flagos:<local-rank>"`) together with `backend="flagos"`. Selecting the
device lazily activates the existing Torch-FL platform adapter; normal CPU and
CUDA/NCCL imports and execution remain unchanged.

Typical `torchrun` code is:

```python
import flagquantum as fq

context = fq.init_torch_distributed(
    device="flagos",
    backend="flagos",
)
```

`device="flagos"` also infers `backend="flagos"`, and `backend="flagos"` with
no device selects the rank-local FlagOS device. Conflicting combinations such
as FlagOS with Gloo, or FlagOS backend with a CPU/CUDA device, fail before a
process group is initialized. An indexed device must also match `LOCAL_RANK`;
FlagQuantum never silently rebinds a mismatched rank-device request.

## Identity and evidence boundary

Every initialized context exposes a machine-readable
`distributed_identity` in `context.summary()`. The F0 schema is
`flagquantum_distributed_identity_v1` and records only facts FlagQuantum can
observe through public APIs:

- outer process-group backend (`flagos`);
- logical rank-local device;
- rank and world size;
- whether a process group is initialized;
- Torch-FL platform identity and version, when provided by the platform API.

F0 deliberately does **not** infer Torch-FL's inner communication route. Until
a provider-owned public identity and collective conformance record are
available, the following fields remain fail-closed:

- `inner_backend=None`;
- `inner_backend_verified=False`;
- `flagcx_route_verified=False`;
- `host_staging_observed=None`;
- `communication_claim_allowed=False`.

`fq.require_verified_flagcx(identity)` therefore raises for every F0-created
FlagOS identity. Merely completing a collective through `backend="flagos"`
does not prove that FlagCX was selected, that data avoided host staging, or
that multi-card correctness/performance has been certified.

## Scope

This change is owned entirely by FlagQuantum. It does not modify Torch-FL or
FlagCX and does not depend on their private Python or native interfaces. It is
control-plane plumbing and contract coverage, not a capability-maturity
promotion. Real single-node and multi-node accelerator tests remain separate
evidence gates.

The unit contract is covered by
`tests/unit/test_flagos_distributed_identity.py`; it uses a mocked public
process-group boundary and makes no hardware or scalability claim.
