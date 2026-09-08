# Platform Providers

This directory is the authoritative home for device discovery, activation,
synchronization, memory, stream/event, RNG, runtime identity, and observed
platform capabilities. Vendor SDK objects stop at this boundary.

It does not choose simulation algorithms, transform programs, schedule jobs,
or promote unverified hardware claims. Runtime selects and orchestrates work;
Simulation owns numerical kernels; Core owns cross-domain contracts.

## Ten-minute change path

- Change portable CPU or CUDA lifecycle behavior in `pytorch.py`.
- Change lazy FlagOS/Torch-FL adaptation in `flagos.py`.
- Change optional FlagGems operator replacement and validation in `flaggems.py`.
- Change provider lookup and device resolution in `registry.py`.
- Change the existing provider-local value types only in `contracts.py`.
- Change CPU-to-Core capability observation in `cpu_target_capabilities.py`.
- Change the observed single-CUDA statevector projection in
  `cuda_target_capabilities.py`; generate evidence with
  `tools/probe_cuda_target_capabilities.py` under one-device visibility.
- Change the deterministic remote-style capability fixture in
  `synthetic_remote.py`; it is test evidence, not a real hardware claim.

Run the platform unit and team tests, followed by the CPU vertical-slice test.
An ordinary platform change should remain inside this directory unless an
existing Core contract genuinely cannot express the required fact.
