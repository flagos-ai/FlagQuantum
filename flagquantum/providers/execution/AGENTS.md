# Execution Provider Team Boundary

This package adapts external QPUs and remote execution services. Keep provider
SDK objects, credentials, submission, status polling, cancellation, and result
decoding behind FlagQuantum-owned contracts. Do not implement simulation
kernels, compiler passes, or Runtime scheduling here.
