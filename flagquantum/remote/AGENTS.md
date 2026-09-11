# Remote Team Boundary

This package adapts compute reached through an external task control plane,
including QPUs, GPU/HPC services, and cloud platforms. Keep external SDK
objects, credentials, submission, status polling, cancellation, and result
decoding behind FlagQuantum-owned contracts. Do not implement direct device
lifecycle, simulation kernels, compiler passes, or Runtime scheduling here.
