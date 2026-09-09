# Jiuding single-GPU adapter handoff — 2026-09-09

Remote team extends the existing experimental JiudingClient with gpus=0 or 1
and optional accelerator_model. No Stable Core export, contract, simulation
kernel, dependency or generic runtime API changes. Scope check passed.

The adapter validates the saved GPU resource configuration before launch.
The worker uses the existing Compute PlatformRuntime for device discovery and
requires exactly one visible CUDA GPU. User scripts explicitly choose CUDA.
The Bell example rejects a CPU result and checks its state numerically.

Default regression: 1237 passed, 14 skipped, 1270 deselected. Focused remote,
local fast path, cloud and Braket regression: 40 passed. Local Python 3.12.14
and torch 2.13.0; Python -S avoids unrelated editable-install path injection.
An initial architecture failure from direct torch.cuda worker calls was fixed
by using the authoritative Compute device discovery; the full suite then passed.

Live single-GPU job aede25bb-ef19-4083-83d3-fb4c06472c49 was accepted with
one A100 requested but remained Pending for 180 seconds. Cancellation was
exercised and terminal Cancelled confirmed. No new resource availability claim.
The final worker separately ran the CUDA Bell example on one visible A100 40GB
in the existing allocated workspace, returning complex64 state error 0.
See ../evidence/jiuding_bell_gpu_20260909.json for these distinct observations.

This does not validate a completed scheduled GPU task, distributed execution,
training or capacity expansion. Shared code, credentials and compatible image
remain prerequisites. No test job was left pending or running.
