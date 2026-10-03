# Remote

Adapt resources reached through an external task control plane: QPUs, GPU/HPC
services, and cloud platforms. The boundary is the submit/status/result
lifecycle; directly controlled devices belong in Compute.

Adapters own credentials, discovery, submission, polling, cancellation, and
result decoding. They translate external data into FlagQuantum-owned contracts.
Compilation, simulation, Runtime policy, and deployment-package semantics stay
with their owning domains.

## Choose an adapter

- [QPU adapters](qpu/): import providers from `flagquantum.remote.qpu` for
  quantum submissions, counts, and provider evidence.
- [Remote compute](compute/README.md): import `JiudingClient` from
  `flagquantum.remote.compute` for batch jobs and reusable Jiuding workspaces.
- [Local target emulation](emulation.py): import `emulate` from
  `flagquantum.remote.emulation` to compile a program for a declared target and
  run the compiled program here, without submitting anything.
- [Quafu guide](../../docs/guides/QUAFU_BACKEND.md): compiler, token, and hardware setup.
- [Jiuding guide](../../docs/guides/JIUDING.md): jobs, workspace execution, and recovery.

## Emulate a target without submitting

`flagquantum.remote.emulation.emulate` takes the same `CloudBackendProfile` an
adapter would submit against. It builds a `TargetCapabilitySnapshot` from that
declaration, runs the target's own legalization, routing, emission, and
conformance passes, executes the compiled program on the local CPU, and returns
the compilation evidence next to the result. It never contacts a provider and is
not re-exported from `flagquantum.remote`, so a caller reaches it by module path
and cannot mistake it for a submission.

The returned record keeps claims that are not the same kind of claim apart:

- the target identity names an emulator, not the device whose profile supplied
  the facts;
- the device kind and precision are observed on this machine, while the qubit
  capacity, native gates, result formats, and limits are the profile's
  declaration, pinned by a digest of its payload;
- the noiseless record is the compiled program's own execution. Passing a
  `noise_model` or a `device_profile` adds a second record that names where its
  channels came from, the representation the planner chose, and whether the
  channel evolution is exact.

```bash
python -m examples.remote.emulate_local_target
```

Noise is opt-in. Without it the result is exact for the compiled program, which
is still a simulation of that program and not of the hardware.

Keep classical compute results distinct from the QPU shots/counts contract.
A provider response must match its submitted program, target, and task; an
ambiguous submission must not be silently retried.

Run offline adapter checks from the repository root:

```bash
python -m pytest tests/test_amazon_braket_provider.py tests/test_cloud_providers.py \
  tests/test_quafu_calibration.py tests/team/remote -q
```

Live acceptance is explicit and target-specific; test doubles do not certify
provider hardware.
