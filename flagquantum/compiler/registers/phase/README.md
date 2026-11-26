# Register phase operations

This package owns phase gradients, programmable rotations, phase costs and measured QFT.

Core, shared Compiler facilities and the register components used by these
implementations are allowed dependencies. Backend selection and numerical
execution belong to Runtime and Simulation. Public APIs and IR schemas follow
their existing change-proposal process.

Start with [phase_gradient.py](phase_gradient.py). Preserve register order, feedback slots and caller-owned
workspace. From the repository root, run the focused scenario tests.

```bash
python -m pytest tests/unit/test_register_phase_gradient.py -q
```

The [register package guide](../README.md) provides executable consumer examples,
source-parity requirements, resource-counting boundaries and the relocation map.

`generalized_qsp.py` interleaves concrete SU2 phases with a bound coherent
unitary's zero/one controls and adjoint, following the pinned GQSP signal order.
Simulation retains numerical phase ownership. The [finite emission guide](../../../../docs/development/GENERALIZED_QSP_EMISSION.md)
gives the append example, workspace obligations, source-stage parity boundary
and separate measured instruction counts versus source component estimates.

The shared [fermionic Fourier pair](fermionic_fourier.py) owns the literal
fifteen-call recipe and exact inverse. Algorithms and interchange constructors
consume this same implementation; caller x/y ports require no scratch or
measurement feedback. Its independent source model lives in the resource owner.
