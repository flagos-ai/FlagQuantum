# Fixed-workload MPS capacity series

Issue #708 is measured with one frozen workload rather than a different circuit
at every rank count: 131,072 sites, bond dimension 768, batch one, complex64,
31 trainable leaves and one Adam step. Logical state size is 1,236,780,012,864
bytes. Every point uses source `4f23154fe9d944ce3f1e801701d59150138b3bd0`
and the unchanged launcher/body digests in `benchmarks/manifests/mps_release_v1.json`.

| Ranks | Nodes | Observed outcome | Maximum rank peak, bytes |
| --- | --- | --- | --- |
| 1 | 1 | CUDA OOM | 84,046,531,584 |
| 2 | 1 | CUDA OOM on every rank | 83,830,173,696 |
| 4 | 1 | CUDA OOM on every rank | 83,830,173,696 |
| 8 | 1 | CUDA OOM on every rank | 83,830,173,696 |
| 16 | 2 | One sharded training and optimizer step completed | 77,745,422,336 |

The [successful hardware run](https://github.com/flagos-ai/FlagQuantum/actions/runs/39895451311)
also passed the existing small training regression on all sixteen ranks before
the capacity sequence. Both hosts used eight A800-SXM4-80GB devices, PyTorch
2.13.0+cu130 and NCCL 2.29.7. The supervised sixteen-rank leg completed in
293.164 seconds; this is a capacity observation, not a matched-speed comparison.
The full sharded forward/backward, gradient ownership, optimizer update and
per-rank memory/communication contract passes the strict per-artifact audit in
`validation.json`.

All five envelopes use keyless runtime-evidence v2. Their checksums cover their
complete provenance and measurements. Checksums establish content integrity;
repository review controls publication. OOM points remain non-release baselines.
Their wire ranges describe intended allocation, not completed training ownership.
The sixteen-rank envelope is a passing candidate, not an automatic promotion.
The series does not certify new topologies or replace the frozen MPS release set.
In particular, the current one-device allocator peak differs from the historical
75,657,923,584-byte baseline pinned in that release manifest; the frozen reference
and the existing v1 HMAC artifacts are unchanged. Full MPS promotion still needs
the declared baseline and matched-speed evidence.

Three failed attempts are retained under `failed-attempts/`. The first omitted
the historical expandable allocator and exhausted memory at sixteen ranks.
The next two reached a 120-second P2P wait limit; the third also passed small
sixteen-rank training with the explicit RDMA rail list. Inspection then found
that the producer omitted the frozen body's explicit `compile_site_kernels=True`:
the runtime's one-step automatic policy selected eager execution. The final
producer restores that frozen option and records an explicit 1,200-second worker
and P2P budget, with a supervising watchdog and thirty-second GPU telemetry.
No workload, precision, truncation budget or scientific acceptance threshold was
relaxed. Allocator and network settings are recorded in `inventory.json`.

Public logs and JSON deterministically replace private infrastructure identities
using node aliases and the existing public-evidence sanitizer before sealing.
Scientific values and source revisions are unchanged. Each envelope binds the
available public log; original file digests are retained separately. The original
run artifacts remain linked above. No old signed envelope was re-sealed.

Diagnostic logs are stored as standard `.log.gz` archives. Decompression retains
their exact deidentified bytes, including whitespace; verify `raw_log_sha256`
against those decompressed bytes. All five provenance log digests are unchanged.

`measurements/world-16-index.json` plus sixteen standard gzip JSON rank records
losslessly reconstruct the deidentified original role document. This retains
the repeated ownership tables without a 36 MB source file. Verify its digest:

```python
import gzip, hashlib, json
from pathlib import Path

root = Path("artifacts/development/mps_capacity_series_20261023")
index = json.loads((root / "measurements/world-16-index.json").read_text())
document = index["document"]
document["ranks"] = [
    json.loads(gzip.decompress((root / item["path"]).read_bytes()))
    for item in index["rank_records"]
]
encoded = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
assert hashlib.sha256(encoded).hexdigest() == index["public_document_sha256"]
```

Reproduce the entire campaign through the existing exclusive accelerator queue:

```bash
gh workflow run scheduled-hardware.yml --ref main \
  -f run_capacity_series=true -f run_multinode=false
python -m pytest tests/benchmark_contract/test_mps_capacity_series_artifacts.py -q
```

For one point, launch `python -m benchmarks.mps_release_evidence --role
capacity-series-point` through `torchrun`, supplying `--local-world-size` and
`--node-count`. All points keep the same release manifest; the sixteen-rank point
also requires its declared `--premise`. Exact executed node commands are retained
under `launches/`; substitute the deidentified interpreter and host addresses.
