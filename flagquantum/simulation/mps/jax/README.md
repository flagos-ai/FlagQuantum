# JAX MPS numerics

This package owns JAX-specific matrix-product-state numerical algorithms. It
does not organize Runtime execution, choose devices, manage distributed
communication, or produce execution evidence.

- Start in `kernels.py` for MPS initialization, gate updates, observables, and
  statevector conversion.
- Start in `batched.py` for batched zero-site creation, site updates and pair
  splitting. Zero-site creation takes the existing batch size and dtype; Runtime
  assigns sites to ranks and selects devices. No Runtime plans enter numerics.
- Start in `pullbacks.py` for local, boundary, and canonicalization pullbacks,
  plus the constrained RY site/Z responses and weighted derivatives. Its
  NumPy development kernels load no JAX; Runtime owns the corresponding
  collectives, timers, optimizer policy and evidence-only allocations.
- Import the owning submodule directly; this package does not re-export a
  facade.
