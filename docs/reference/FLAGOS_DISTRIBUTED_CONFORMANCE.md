# FlagOS distributed conformance

FlagQuantum's single-node FlagOS conformance harness validates the public
Torch-FL ProcessGroup boundary with real accelerator tensors and one genuinely
sharded statevector workload. It is the next evidence layer after the
[F0 control-plane contract](FLAGOS_DISTRIBUTED_F0.md).

The first measured A800 result and its fail-closed decision are recorded in
[FlagOS distributed F1 result on A800](FLAGOS_DISTRIBUTED_F1_A800_RESULT.md).
The FlagQuantum-only precision remediation and scoped workload result are
recorded in
[FlagOS distributed F1.1 result on A800](FLAGOS_DISTRIBUTED_F11_A800_RESULT.md).
The next 2/4/8-card forward scale layer is specified in
[FlagOS statevector F2 single-node scale profile](FLAGOS_STATEVECTOR_F2_SCALE_PROFILE.md).

Run it in an explicitly provisioned Torch-FL environment:

```bash
torchrun --standalone --nproc-per-node=2 \
  tools/validate_flagos_distributed_conformance.py \
  --output flagos-distributed-conformance.json
```

The harness imports Torch-FL before PyTorch, initializes each rank with
`device="flagos"` and `backend="flagos"`, and enforces a one-to-one mapping
between rank, local rank, and logical device.

## Mechanical coverage

Both `complex64` and `complex128` must pass every required tensor route:

- `broadcast`;
- `all_reduce`;
- `all_gather_into_tensor`;
- `reduce_scatter_tensor`;
- `isend`/`irecv` through `batch_isend_irecv`.

Python object collectives are replaced with fail-fast guards during the run.
Every collective payload and output must remain on the logical `flagos` device.
This proves logical tensor residency at the public ProcessGroup boundary; it
does not inspect provider-native buffers or prove the absence of staging inside
Torch-FL or its native communication implementation.

The harness also executes a three-wire statevector containing gates on the
rank-address-sharded wire. Each process retains only its strict amplitude shard
and compares that shard with the matching indices from a bounded CPU reference.
No distributed result reconstructs or gathers the full state. The report must
record positive distributed-gate, communication-count, and communication-byte
measurements.

## Machine-readable boundary

The output schema is `flagquantum_flagos_distributed_conformance_v1`. A passing
payload records:

- complete rank placement;
- all ten collective/dtype checks;
- complex64 and complex128 sharded-statevector checks;
- FlagQuantum `DistributedIdentity`;
- environment and source revision;
- explicit blockers and claim permissions.

The harness also writes the same schema and returns a non-zero exit status when
the runtime rejects a collective or a numerical check fails. Each failed
collective retains its public exception in `error`, so an unsupported complex
dtype is evidence rather than a lost subprocess traceback.

The report separates `mechanical_conformance_accepted`, which requires the
entire collective matrix, from `statevector_workload_conformance_accepted`,
which requires both complex dtypes, true sharding, and the tensor collectives
actually exercised by the production statevector route. The narrower status
never removes a failed collective and never enables a communication, FlagCX,
scalability, release, or production claim.

`status="passed"` means only that the mechanical FlagOS ProcessGroup contract
passed. FlagQuantum neither requires a Torch-FL process-group identity API nor
infers an inner provider from successful collectives. The same payload must
therefore retain:

```text
flagcx_route_verified = false
host_staging_observed = null
communication_claim_allowed = false
scalability_claim_allowed = false
release_gate_allowed = false
```

Consequently this harness cannot certify FlagCX, domestic hardware, performance,
capacity scaling, or production support. It changes no capability-maturity
level.

## Tests

The pure contract is covered without hardware by
`tests/unit/test_flagos_distributed_conformance.py`. The two-device integration
test is opt-in:

```bash
FLAGQUANTUM_TEST_FLAGOS_DISTRIBUTED=1 \
  pytest tests/test_flagos_distributed_conformance.py -v
```

All implementation and evidence code lives in FlagQuantum. The harness neither
imports FlagCX nor accesses Torch-FL private objects, and it requires no change
or pull request in either external repository.
