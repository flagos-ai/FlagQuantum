# Lazy Torch training boundary review evidence

The user approved the narrow #936 boundary on 2026-11-02. See
`docs/development/TORCH_TRAINING_COMPILATION_CONTRACT.md` for exact import
permission, unchanged API/defaults, ownership, exclusions and reserved evaluation
cases. This packet is contract conformance, not a performance result or closure
of #936/#918. Production implementation follows the merged contract.

The nine replacement-fake cases use an identity Torch compiler backend, including
one direct `fq.train` consumer. They preserve independent analytic loss/VJPs,
six-step SGD/Adam trajectories, changing trainable input batches 1/8, user hooks,
no-grad evaluation and fresh-Module 3+3 checkpoint recovery. Only eligible SGD
uses the compiler fake; hooked or Adam cases retain explicit eager reasons.
Together with 16 architecture checks, 25 tests passed locally. The real
`fq.train` capture warns that SHA-256 is not traceable; graph breaks remain
visible. Identity backend and fragmented graphs must not be called numerical
code generation or whole-training fusion.

The independent falsification bypasses the Adam exclusion. On CPU Torch 2.14.1,
exact float64, six updates, the identity compiler preserves SGD's parameter and
loss trajectories exactly. Adam's final parameter maximum difference is
approximately 1.2863253e-6. Both trajectories, losses and capture counts are
retained in `optimizer-capture.json`; this is an explicit blocker for compiled
Adam, even without numerical code generation. It does not diagnose the upstream
root cause or certify behavior in other Torch versions.

The numeric producer is the clean commit recorded in `source_commit`; its
reproducer and fake hashes are included. `publication.json` hashes sanitized
deterministic-gzip logs and receipts. No checkpoint binaries are retained. Run:

```bash
python artifacts/development/torch_training_contract_20261102/verify.py
python -m pytest tests/unit/test_training_compilation_contract_fake.py tests/unit/test_architecture_boundaries.py -q
PYTHONPATH=. python artifacts/development/torch_training_contract_20261102/reproduce_optimizer_capture.py --output /tmp/optimizer-capture-new.json
```

Use the recorded producer checkout and matching Torch version to reproduce the
retained numbers. Other versions may have different compilation behavior; retain
their outcomes rather than rewriting this historical receipt. The fake tests do
not enable compilation in the production training loop.
