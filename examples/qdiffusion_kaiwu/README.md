# QDiffusion with Kaiwu

This directory separates development checks from acceptance evidence.

`a800_sampler_smoke.py` is a development probe. It uses the real Kaiwu PyTorch
Plugin data path and an explicitly selected in-memory fake transport. It can
verify A800 tensor placement, matrix and sample transfers, backward, and an
optimizer update. It cannot establish QBoson hardware use or QDiffusion system
acceptance.

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
