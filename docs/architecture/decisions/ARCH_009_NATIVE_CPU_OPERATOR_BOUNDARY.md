# ARCH-009: Package-local native CPU operator boundary

Status: Approved
Date: 2026-09-28
Scope: internal Simulation implementation and package build; no Stable Core change

## Context

The single-device statevector adjoint path spends most of its backward time in
many small PyTorch tensor expressions and their intermediate copies. The RX,
RY, and RZ reverse steps all traverse the same amplitude pairs three times:
once for the parameter VJP, once for the ket, and once for the adjoint. Python
composition cannot fuse those traversals reliably on CPU.

FlagQuantum also needs an operator layout that can later host CPU, GPU, and
distributed communication kernels without putting numerical implementation in
Runtime or creating one extension module per optimization.

## Decision

- Native numerical sources live below the owning Simulation package in
  `flagquantum/simulation/native_cpu/csrc/`.
- One private extension module, `flagquantum.simulation.native_cpu._C`, registers
  internal operators through the PyTorch dispatcher.
- Runtime may select the operator, but numerical implementation remains owned
  by Simulation.
- The initial operators fuse RX/RY/RZ/RZZ parameter reduction with inverse ket
  and adjoint updates for contiguous complex64/complex128 CPU statevectors.
  Contiguous commuting RZZ regions are reduced in one state traversal, including
  an aggregate fast path for adjacent gates that share one parameter.
- The extension is built with the distribution. Runtime JIT compilation is not
  permitted.
- Every native path retains a PyTorch fallback. `FQ_NATIVE_CPU_ADJOINT=0`
  provides an explicit operational rollback.
- Unsupported gates, layouts, devices, and dtypes return to the existing path;
  the native module does not expand public support claims.

Future distributed or device-native operators may follow the same ownership
pattern, but they require their own evidence and admission decision. This ADR
does not authorize a distributed capability claim.

## Consequences

Binary wheels become platform- and Python-specific and source builds require a
compatible C++ compiler plus PyTorch headers. In return, installed wheels do
not compile code on first use, the hot loop uses PyTorch's CPU thread pool, and
the Python public API remains unchanged. Distribution reproducibility and
clean-wheel/source-install checks remain required CI gates.

## Approval record

The repository owner authorized implementation of the native fused CPU adjoint
kernel and this package-local organization on 2026-09-28. Approval is limited
to the internal vertical slice described above.
