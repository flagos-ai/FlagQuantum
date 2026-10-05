# QDiffusion with Kaiwu

This directory separates development checks from acceptance evidence.

`a800_sampler_smoke.py` is a development probe. It uses the real Kaiwu PyTorch
Plugin data path and an explicitly selected in-memory fake transport. It can
verify A800 tensor placement, matrix and sample transfers, backward, and an
optimizer update. It cannot establish QBoson hardware use or QDiffusion system
acceptance.

The development probe records both the stable validation-host alias and the
machine-reported hostname because they are different on the current systems.
The container hostname is explicitly set to the observed host identity because
Docker otherwise assigns an unrelated container ID. After extracting the
reviewed source and plugin archives, invoke the bounded runner on the first host
as follows:

```bash
mkdir -m 700 private-evidence
bash examples/qdiffusion_kaiwu/run_a800_development_probe.sh \
  jp-a800-171 \
  bm-baai-dx-zone1-lc-a800-80g-15-171 \
  "$PWD" \
  /absolute/path/to/kaiwu-pytorch-plugin \
  "$PWD/private-evidence/jp-a800-171-development.json" \
  045e54276ef86d96524a39486fea93617952fb39 \
  f047bce7b1077449967bbe9e9fab5741542b48d4
```

The runner executes `qdiffusion_system_development_probe.py`, covering proposal
forward, conditioned Boltzmann sampling through `KaiwuSampler`, energy
objective, backward, optimizer update, and one guided generation step. The
output is a new mode-0600 file and is never an acceptance record because the
transport is explicitly the in-memory development fake. The runner exposes
only GPU 0, disables networking, mounts both code trees read-only, uses a
read-only container filesystem, and persists only the requested evidence file.

The acceptance lane uses a frozen configuration and two independent host
records:

1. Copy `acceptance_config.example.json` to an evidence directory and replace
   every placeholder before any baseline or guided experiment runs.
2. Record the file's SHA-256 digest in each host record and in a manifest based
   on `acceptance_manifest.example.json`.
3. Run the bounded system path independently on `jp-a800-171` and
   `jp-a800-172`. The configured primary host also runs every frozen seed in the
   protein experiment; the replay host runs the fixed portability fixture.
4. Validate the finished bundle:

```bash
python examples/qdiffusion_kaiwu/validate_acceptance.py path/to/manifest.json
```

The validator recomputes the decision from evidence. It requires an observed
A800 on both hosts, independent QBoson task identities, `kaiwu_cim` transport,
real-provider evidence, no fallback or retrieval resubmission, bounded calls,
finite training values, valid generation, and the preregistered application
thresholds. A record that uses the development fake, CPU tensors, one host, or
post-hoc metric thresholds fails closed.

Credentials never belong in the frozen configuration or evidence bundle.

## Live provider smoke test

`qboson_live_smoke.py` is a separately invoked, quota-consuming Phase 2 probe.
It submits one fixed optimization task and one fixed sampling task, uses the
same identity for bounded polling, and writes a new mode-0600 record without
credentials or raw vendor exception text. It has no simulator fallback.

The command requires `QBOSON_USER_ID`, `QBOSON_SDK_CODE`, an existing private
checkpoint directory, and an explicitly selected project. The acknowledgement
must be typed exactly so an ordinary test run cannot spend provider quota:

```bash
python examples/qdiffusion_kaiwu/qboson_live_smoke.py \
  --checkpoint-dir private-kaiwu-checkpoints \
  --output private-evidence/qboson-smoke.json \
  --project-no CPQC-your-project \
  --task-prefix flagquantum-smoke-20261005 \
  --acknowledge-provider-cost I_ACKNOWLEDGE_QBOSON_QUOTA_USAGE
```

Successful tasks alone do not make this an acceptance record. The script keeps
`hardware_acceptance=false` until the pinned SDK mapping supplies both a stable
provider task ID and a provider-reported target for every task.
