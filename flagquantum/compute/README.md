# Compute

`flagquantum.compute` is the authoritative home for compute resources that the
current process controls directly. It owns device discovery, activation,
synchronization, memory, stream/event, RNG, runtime identity, and observed
hardware capabilities. Vendor runtime objects stop at this boundary.

It does not choose simulation algorithms, transform programs, schedule jobs,
or promote unverified hardware claims. Runtime selects and orchestrates work;
Simulation owns numerical kernels; Core owns cross-domain contracts.

## Ten-minute change path

- Change portable CPU or CUDA lifecycle behavior in `pytorch.py`.
- Change lazy FlagOS/Torch-FL adaptation in `flagos.py`.
- Change optional FlagGems operator replacement and validation in `flaggems.py`.
- Change compute lookup and device resolution in `registry.py`.
- Change the existing compute-local value types only in `contracts.py`.
- Change CPU-to-Core capability observation in `cpu_target_capabilities.py`.
- Change the observed single-CUDA statevector projection in
  `cuda_target_capabilities.py`; generate evidence with
  `tools/probe_cuda_target_capabilities.py` under one-device visibility.
Run the platform unit and team tests, followed by the CPU vertical-slice test.
An ordinary compute-platform change should remain inside this directory unless an
existing Core contract genuinely cannot express the required fact.
