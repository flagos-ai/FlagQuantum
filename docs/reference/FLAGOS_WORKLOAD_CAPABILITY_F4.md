# FlagOS workload capability matrix (F4)

F4 turns the checked-in F1, F2, and F3 hardware records into one machine-readable
answer about workloads that have been demonstrated through FlagQuantum's public
`backend="flagos"` and `device="flagos"` boundary. It does not inspect or infer
the provider's inner communication route.

## Current result

The authoritative F4 artifact is
[`artifacts/flagos_workload_capability_f4_20260826.json`](../../artifacts/flagos_workload_capability_f4_20260826.json).
It classifies the fixed evidence set as follows:

| Capability | Status | Exact evidence boundary |
| --- | --- | --- |
| `flagos.statevector.forward` | `development_verified` | complex64 and complex128, one node, 2/4/8 cards, sharded statevector |
| `flagos.statevector.training` | `development_verified` | complex64 and complex128 bounded backward, SGD, and Adam trajectories, one node, 2/4/8 cards |
| `flagos.collectives.full` | `unsupported` | complex `reduce_scatter_tensor` failed on the two-card F1 matrix; the other required collectives passed |

`development_verified` means executable development hardware evidence exists for
the stated workload and nothing broader. It is not production support, general
scalability evidence, a performance claim, or a multi-node result. `unsupported`
describes the tested full collective matrix; it does not prevent statevector
workloads whose required collective subset passed.

## Audit contract

Run the deterministic aggregator from the repository root:

```bash
python tools/audit_flagos_workload_capability.py
```

The tool hashes each source artifact, validates raw cases and rank semantics, and
then writes the F4 artifact atomically. It fails closed if schemas, 2/4/8-card
ladders, sharding, gradient or optimizer semantics, source revisions, provider
claim flags, or explicit unsupported results disagree.

The output deliberately records:

- `outer_backend = "flagos"`;
- `inner_communication_route = "unattributed"`;
- `flagcx_route_verified = false`;
- all communication, scalability, production, and release claim flags as false.

The historical source artifacts may contain older diagnostic wording. F4 does
not propagate that wording into a provider-identity claim.

## Evidence inputs

- F1.1 two-card distributed conformance, including the complex collective matrix;
- F2 one-node 2/4/8-card sharded forward scale ladder;
- F3 one-node 2/4/8-card sharded backward and optimizer ladder.

All three inputs use the same locked Torch-FL source revision. Their individual
FlagQuantum source revisions remain visible in the artifact descriptors.

F4 is a historical aggregation of those three inputs, so its
`single_device_capacity_failure_not_measured` blocker remains unchanged. The
later matched 32-qubit capacity result is recorded separately in
[F5](FLAGOS_STATEVECTOR_CAPACITY_F5.md); it does not retroactively rewrite the
F4 artifact.
