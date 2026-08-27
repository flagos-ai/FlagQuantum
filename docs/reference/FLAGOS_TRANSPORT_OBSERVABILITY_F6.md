# FlagOS transport observability (F6)

F6 adds a fail-closed observation contract around the distributed collectives
used through FlagQuantum's public `backend="flagos"` and `device="flagos"`
boundary. It records what the runtime and profiler actually expose; it does not
infer the provider's inner transport from successful collective execution.

The authoritative artifact is
[`artifacts/flagos_transport_observability_f6_a800_20260827.json`](../../artifacts/flagos_transport_observability_f6_a800_20260827.json),
with SHA-256
`0a238e869e1219f2b1924e986305c67ba8ba09a0d55d2a03ebba1f64130db491`.

## Measured result

The single-node A800 ladder exercised world sizes 2, 4, and 8. Every rank ran
`all_gather_into_tensor`, `all_reduce`, `broadcast`, and `isend`/`irecv` with
both complex64 and complex128 tensors. All 112 rank-local observations passed,
kept their inputs and outputs on the logical `flagos` device, and reported zero
maximum absolute error.

This is valid **FlagOS distributed observation evidence** for the tested
matrix. It is not FlagCX route evidence.

## Profiler boundary

The profiler was requested with CPU and CUDA activities, but CUPTI did not
produce a usable device timeline in the multi-process run. Each bounded
collective interval retained the CPU-side `aten::view_as_real` event with zero
device time. The artifact therefore records:

- `profiler_capture_complete = false`;
- `device_activity_observed_all = false`;
- `host_staging_observed = null`;
- `no_host_staging_certified = false`.

No explicitly directional HtoD or DtoH memcpy event was observed. That absence
is not proof that host staging did not occur because the device-side capture is
incomplete. Undirected names such as `aten::copy_` or `cudaMemcpyAsync` would
also be insufficient: only an event name with an explicit host/device
direction is classified as a positive host-transfer observation.

## Route attribution boundary

FlagQuantum does not inspect private ProcessGroup fields and does not require a
nonexistent provider API such as `torch_fl.distributed_identity(group)`.
Without a provider-owned attestation contract, the frozen evidence keeps:

- `outer_backend = "flagos"`;
- `inner_communication_route = "unattributed"`;
- `flagcx_route_verified = false`;
- communication, scalability, production, and release claims disabled.

This separation lets FlagQuantum improve its own observability without taking
ownership of Torch-FL or FlagCX internals. Direct FlagCX attribution remains a
future provider-coordination task, not something inferred by this artifact.

## Environment and reproduction

The run used FlagQuantum revision
`4f4158f6250f9f6bf17cac7ae565eb5bf106cee1`, Torch-FL revision
`1b19383c24d9aa54637dd23557a5a20ef6d52ec5`, PyTorch `2.11.0+cu130`, CUDA
13.0, and eight NVIDIA A800-SXM4-80GB devices. Torch-FL was built in an
isolated CUDA-boxing reference environment with FlagGems disabled.

On an isolated eight-device FlagOS environment, run:

```bash
python tools/observe_flagos_transport.py \
  --world-sizes 2 4 8 \
  --output artifacts/flagos_transport_observability_f6_a800_20260827.json
```

The controller launches a separate distributed job for each world size and
aggregates rank-local records. The checked-in benchmark-contract test rebuilds
the top-level profile from those records and rejects missing matrix cells,
rank or revision drift, device-residency failures, forged transfer summaries,
and route or no-staging overclaims.
