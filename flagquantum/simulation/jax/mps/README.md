# JAX MPS numerics

This package owns JAX-specific matrix-product-state numerical algorithms. It
does not organize Runtime execution, choose devices, manage distributed
communication, or produce execution evidence.

- Start in `kernels.py` for MPS initialization, gate updates, observables, and
  statevector conversion.
- Import the owning submodule directly; this package does not re-export a
  facade.
